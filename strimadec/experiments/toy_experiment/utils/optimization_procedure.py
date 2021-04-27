import torch
import numpy as np

from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR, RELAX


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
                temp (float): temperature of concrete distribution in log units [1]
            estimator_name == "NVIL":
                baseline_net (nn.Sequential): neural baseline network
            estimator_name == "REBAR":
                eta (torch tensor): tuneable hyperparameter that scales control variate [1]
                log_temp (torch tensor): tuneable hyperparameter (temperature of concrete) [1]
            estimator_name == "RELAX":
                c_phi (nn.Sequential): neural network that takes relaxed & conditioned relaxed input

    Returns:
        results (dict): containing the following metrics
            expected_losses (numpy array): expected loss for each epoch [num_epochs]
            vars_grad (numpy array): variance of gradient estimator per epoch [num_epochs]
    """
    # make optimization procedure reproducible
    torch.manual_seed(params["SEED"])
    # retrieve independent params
    x, target, encoder_net = params["x"], params["target"], params["encoder_net"]
    loss_func = params["loss_func"]
    estimator_name = params["estimator_name"]
    # push to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, target, encoder_net = x.to(device), target.to(device), encoder_net.to(device)
    # define optimzer for encoder_net
    optimizer = torch.optim.Adam(encoder_net.parameters(), lr=params["lr"])
    # define tuneable_params (if they exist) based on estimator_name
    if estimator_name in ["Exact gradient", "REINFORCE"]:
        tuneable_params = []
    elif estimator_name == "CONCRETE":
        temp = params["temp"].to(device)
    elif estimator_name == "NVIL":
        baseline_net = params["baseline_net"].to(device)
        tuneable_params = baseline_net.parameters()
    elif estimator_name == "REBAR":
        eta, log_temp = params["eta"], params["log_temp"]
        eta = eta.to(device).detach().requires_grad_(True)
        log_temp = log_temp.to(device).detach().requires_grad_(True)
        tuneable_params = [eta, log_temp]
    elif estimator_name == "RELAX":
        c_phi = params["c_phi"].to(device)
        tuneable_params = c_phi.parameters()
    # define tuneable_params optimizer (if they exist)
    if tuneable_params:
        tune_optimizer = torch.optim.Adam(tuneable_params, lr=params["tune_lr"])
    # track expected_losses and gradient variances
    expected_losses = np.zeros([params["num_epochs"]])
    num_classes = target.shape[1]
    vars_grad = np.zeros([params["num_epochs"], num_classes])
    # start training procedure
    for epoch in range(params["num_epochs"]):
        optimizer.zero_grad()
        if tuneable_params:
            tune_optimizer.zero_grad()

        # get probability vector of categorical distribution in logits [1, L]
        probs_logits = encoder_net.forward(x)
        # upsample logits and target to [batch, L]
        probs_logits_ups = probs_logits.repeat(params["batch_size"], 1)
        target_ups = target.repeat(params["batch_size"], 1)
        # compute estimator through which we can backpropagate
        if params["estimator_name"] == "Exact gradient":
            estimator = analytical(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "REINFORCE":
            estimator = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "NVIL":
            baseline_vals_ups = baseline_net.forward(x).repeat(params["batch_size"], 1)
            estimator = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif params["estimator_name"] == "CONCRETE":
            estimator = CONCRETE(probs_logits_ups, target_ups, temp, loss_func)
        elif params["estimator_name"] == "REBAR":
            temp = log_temp.exp()
            estimator = REBAR(probs_logits_ups, target_ups, temp, eta, params["loss_func"])
        elif params["estimator_name"] == "RELAX":
            estimator = RELAX(probs_logits_ups, target_ups, c_phi, loss_func)
        estimator.backward()

        optimizer.step()
        if tuneable_params:
            tune_optimizer.step()

        ################## TRACK METRICS ########################
        # expected loss
        probs = torch.softmax(probs_logits, 1)
        expected_loss = 0
        for class_ind in range(num_classes):
            # define one hot class vector corresponding to class_ind [1, L]
            one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0).to(device)
            # compute loss corresponding to class_ind
            cur_loss = loss_func(one_hot_class, target).mean()
            expected_loss += probs[:, class_ind] * cur_loss
        expected_losses[epoch] = expected_loss
        # variance of gradient estimator (upsample by FIXED_BATCH for useful var estimator)
        optimizer.zero_grad()
        if tuneable_params:
            tune_optimizer.zero_grad()
        x_ups = x.repeat(params["FIXED_BATCH"], 1)
        target_ups = target.repeat(params["FIXED_BATCH"], 1)
        probs_logits_ups = encoder_net.forward(x_ups)
        probs_logits_ups.retain_grad()
        if params["estimator_name"] == "Exact gradient":
            estimator_ups = analytical(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "REINFORCE":
            estimator_ups = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "NVIL":
            baseline_vals_ups = baseline_net.forward(x_ups)
            estimator_ups = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif params["estimator_name"] == "CONCRETE":
            estimator = CONCRETE(probs_logits_ups, target_ups, temp, loss_func)
        elif params["estimator_name"] == "REBAR":
            temp = log_temp.exp()
            estimator_ups = REBAR(probs_logits_ups, target_ups, temp, eta, params["loss_func"])
        elif params["estimator_name"] == "RELAX":
            estimator_ups = RELAX(probs_logits_ups, target_ups, c_phi, loss_func)
        estimator_ups.backward()
        # retrieve gradient of estimator [FIXED_BATCH, L]
        g_estimator = probs_logits_ups.grad
        for class_ind in range(num_classes):
            vars_grad[epoch, class_ind] = g_estimator.var(dim=0)[class_ind].item()
    if params["estimator_name"] == "Exact gradient":
        # make sure that analytical estimator converges to true optimum by computing optimal loss
        possible_losses = torch.zeros(num_classes)
        for class_ind in range(num_classes):
            one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0).to(device)
            possible_losses[class_ind] = loss_func(one_hot_class, target).mean()
        optimal_loss = min(possible_losses)
        assert (optimal_loss - expected_loss) ** 2 < 1e-6, "analytical solution seems to be wrong"
    results = {"expected_losses": expected_losses, "vars_grad": vars_grad.sum(1)}
    return results