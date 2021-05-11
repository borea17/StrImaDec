import time

import torch
import numpy as np
from pytorch_lightning import seed_everything

from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR, RELAX
from strimadec.discrete_gradient_estimators.utils import set_requires_grad


def run_stochastic_optimization(params):
    """
        execute the stochastic optimization for a specific setup defined in the params dict

    Args:
        params (dict): dictionary containing all necessary variables (vary depending on estimator)
            ################## INDEPENDENT PARAMS ##################
            x (torch tensor): input of neural network [1, x_dim]
            target (torch tensor): defines the target in the stochastic optimization problem [1, L]
            encoder_net (nn.Sequential): is used to retrieve the logits probabilities
            num_epochs (int): number of iterations during training
            batch_size (int): effectively the number of samples used for Monte-Carlo integration
            lr (float): learning rate for encoder_net
            loss_func (method): loss function using sampled class and a target
            estimator_name (str): defines the gradient estimator that should be applied
            FIXED_BATCH (int): used to obtain a good estimate for the variance of the estimator
            ################## DEPENDENT PARAMS ##################
            estimator_name == "CONCRETE":
                temp (float): temperature of concrete distribution
            estimator_name == "NVIL":
                baseline_net (nn.Sequential): neural baseline network
            estimator_name == "REBAR":
                eta (float): hyperparameter that scales control variate [1]
                log_temp (nn.Parameter): tuneable hyperparameter (temperature of concrete) [1]
            estimator_name == "RELAX":
                c_phi (nn.Sequential): neural network that takes relaxed & conditioned relaxed input

    Returns:
        results (dict): containing the following metrics
            expected_losses (numpy array): expected loss for each epoch [num_epochs]
            vars_grad (numpy array): variance of gradient estimator per epoch [num_epochs]
    """
    # make optimization procedure reproducible
    seed_everything(params["SEED"])
    # retrieve independent params
    x, target, encoder_net = params["x"], params["target"], params["encoder_net"]
    loss_func = params["loss_func"]
    estimator_name = params["estimator_name"]
    # push to cuda if available (rather for testing than for efficiency)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x, target, encoder_net = x.to(device), target.to(device), encoder_net.to(device)
    # define optimzer for encoder_net
    optimizer = torch.optim.Adam(encoder_net.parameters(), lr=params["lr"])
    # define tuneable_params (if they exist) based on estimator_name
    if any(est in estimator_name for est in ["REINFORCE", "Exact gradient", "CONCRETE"]):
        tuneable_params = []
    elif "NVIL" in estimator_name:
        baseline_net = params["baseline_net"].to(device)
        tuneable_params = baseline_net.parameters()
    elif "REBAR" in estimator_name:
        eta, log_temp = params["eta"], params["log_temp"]
        # simple `.to(device)` causes error on cuda due to `is_leaf` becoming `False`
        log_temp = log_temp.to(device).detach().requires_grad_(True)
        model = encoder_net
        tuneable_params = [log_temp]
    elif "RELAX" in estimator_name:
        c_phi = params["c_phi"].to(device)
        model = encoder_net
        tuneable_params = c_phi.parameters()

    if tuneable_params:  # define tuneable_params optimizer (if they exist)
        tune_optimizer = torch.optim.Adam(tuneable_params, lr=params["tune_lr"])
    # track expected_losses, gradient variances and computation times
    expected_losses = np.zeros([params["num_epochs"]])
    num_classes = target.shape[1]
    vars_grad = np.zeros([params["num_epochs"], num_classes])
    elapsed_times = np.zeros([params["num_epochs"]])
    # start training procedure
    for epoch in range(params["num_epochs"]):
        tic = time.time()

        optimizer.zero_grad()
        if tuneable_params:
            tune_optimizer.zero_grad()

        # get probability vector of categorical distribution in logits [1, L]
        probs_logits = encoder_net.forward(x)
        # upsample logits and target to [batch, L]
        probs_logits_ups = probs_logits.repeat(params["batch_size"], 1)
        target_ups = target.repeat(params["batch_size"], 1)
        # compute estimator through which we can backpropagate
        if "Exact gradient" in estimator_name:
            estimator = analytical(probs_logits_ups, target_ups, loss_func)
        elif "REINFORCE" in estimator_name:
            estimator, _ = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif "NVIL" in estimator_name:
            baseline_vals_ups = baseline_net.forward(x).repeat(params["batch_size"], 1)
            estimator, _ = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif "CONCRETE" in estimator_name:
            estimator = CONCRETE(probs_logits_ups, target_ups, params["temp"], loss_func)
        elif "REBAR" in estimator_name:
            temp = log_temp.exp()
            estimator, _ = REBAR(probs_logits_ups, target_ups, temp, eta, model, loss_func)
            log_temp.requires_grad = False
        elif "RELAX" in estimator_name:
            estimator, _ = RELAX(probs_logits_ups, target_ups, c_phi, model, loss_func)
            set_requires_grad(c_phi, False)  # do not update c_phi parameters in estimator backward

        estimator.sum().backward()

        optimizer.step()
        if tuneable_params:
            if "RELAX" in estimator_name:
                set_requires_grad(c_phi, True)  # update c_phi when calling tune optimizer step
            elif "REBAR" in estimator_name:
                log_temp.requires_grad = True
            tune_optimizer.step()
        ################## TRACK METRICS ########################
        ### computation times in seconds ###
        elapsed_time = time.time() - tic
        elapsed_times[epoch] = elapsed_time
        ### expected loss ###
        # get probs from probs_logits [1, L]
        probs = torch.softmax(probs_logits, 1)
        # get batch_size one-hot vectors [1, L, L]
        one_hot_vectors = torch.eye(probs.shape[1]).unsqueeze(0).to(probs.device)
        # compute loss for each one_hot_vector [1, L]
        loss_per_one_hot = params["loss_func"](one_hot_vectors, target.unsqueeze(1))
        # compute expected loss by multiplying with probs
        expected_loss = (probs * loss_per_one_hot).sum()
        expected_losses[epoch] = expected_loss.item()
        ### variance of gradient estimator (upsample by FIXED_BATCH for useful var estimator) ###
        x_ups = x.repeat(params["FIXED_BATCH"], 1)
        target_ups = target.repeat(params["FIXED_BATCH"], 1)
        probs_logits_ups = encoder_net.forward(x_ups)
        probs_logits_ups.retain_grad()
        if "Exact gradient" in estimator_name:
            estimator_ups = analytical(probs_logits_ups, target_ups, loss_func)
        elif "REINFORCE" in estimator_name:
            estimator_ups, _ = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif "NVIL" in estimator_name:
            baseline_vals_ups = baseline_net.forward(x_ups)
            estimator_ups, _ = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif "CONCRETE" in estimator_name:
            estimator_ups = CONCRETE(probs_logits_ups, target_ups, params["temp"], loss_func)
        elif "REBAR" in estimator_name:
            temp = log_temp.exp()
            estimator_ups, _ = REBAR(probs_logits_ups, target_ups, temp, eta, model, loss_func)
        elif "RELAX" in estimator_name:
            estimator_ups, _ = RELAX(probs_logits_ups, target_ups, c_phi, model, loss_func)

        estimator_ups.sum().backward()
        # retrieve gradient of estimator [FIXED_BATCH, L]
        g_estimator = probs_logits_ups.grad
        for class_ind in range(num_classes):
            vars_grad[epoch, class_ind] = g_estimator.var(dim=0)[class_ind].item()
    if "Exact gradient" in estimator_name:
        # make sure that analytical estimator converges to true optimum by computing optimal loss
        possible_losses = torch.zeros(num_classes)
        for class_ind in range(num_classes):
            one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0).to(device)
            possible_losses[class_ind] = loss_func(one_hot_class, target)
        optimal_loss = min(possible_losses)
        assert (optimal_loss - expected_loss) ** 2 < 1e-6, "analytical solution seems to be wrong"
    results = {
        "expected_losses": expected_losses,
        "vars_grad": vars_grad.sum(1),
        "elapsed_times": elapsed_times,
    }
    return results