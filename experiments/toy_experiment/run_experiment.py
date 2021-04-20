import torch
import torch.nn as nn
import numpy as np

from strimadec.discrete_gradient_estimators import REINFORCE


def run_experiment():
    estimator_names = ["REINFORCE"]
    for estimator_name in estimator_names:
        params = build_experimental_setup(estimator_name)
        results = run_stochastic_optimization(params)
    return


def build_experimental_setup(estimator_name):
    """
        creates the experimental setup params given the estimator_name

    Args:
        estimator_name (str): name of gradient estimator

    Returns:
        params (dict): dictionary that can be used to execute
            `run_stochastic_optimization`
    """
    x = torch.ones([1, 1])
    target = torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0)
    encoder_net = nn.Sequential(nn.Linear(1, 3), nn.ReLU())

    params = {
        "SEED": 42,
        "x": x,
        "target": target,
        "encoder_net": encoder_net,
        "num_epochs": 1,
        "batch_size": 1,
        "lr": 0.01,
        "loss_func": lambda x, y: (x - y) ** 2,
        "estimator_name": estimator_name,
        "FIXED_BATCH": 1000,
    }
    return params


def run_stochastic_optimization(params):
    """
        execute the stochastic optimization for a specific setup defined in params

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
                log_temp (torch tensor): temperature of concrete distribution in log units [1]
            estimator_name == "NVIL":
                baseline_net (nn.Sequential): neural baseline network

    Returns:
        results (dict): containing the following metrics
            expected_losses (numpy array): expected loss for each epoch [num_epochs]
            vars_grad (numpy array): variance of gradient estimator per epoch [num_epochs]
    """
    torch.manual_seed(params["SEED"])
    # retrieve independent params
    x, target, encoder_net = params["x"], params["target"], params["encoder_net"]
    loss_func = params["loss_func"]
    estimator_name = params["estimator_name"]
    # define optimzer for encoder_net
    optimizer = torch.optim.Adam(encoder_net.parameters(), lr=params["lr"])
    # define tuneable_params (if they exist) based on estimator_name
    if estimator_name in ["REINFORCE", "CONCRETE"]:
        tuneable_params = []
    elif estimator_name == "NVIL":
        baseline_net = params["baseline_net"]
        tuneable_params = baseline_net.parameters()
    elif estimator_name == "REBAR":
        eta, log_temp = params["eta"], params["log_temp"]
        tuneable_params = [eta, log_temp]
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
        if params["estimator_name"] == "REINFORCE":
            estimator = REINFORCE(probs_logits_ups, target_ups, loss_func)
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
            one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0)
            # compute loss corresponding to class_ind
            cur_loss = loss_func(one_hot_class, target).mean()
            expected_loss += probs[:, class_ind] * cur_loss
        expected_losses[epoch] = expected_loss
        # variance of gradient estimator (upsample by FIXED_BATCH for useful var estimator)
        x_ups = x.repeat(params["FIXED_BATCH"], 1)
        target_ups = target.repeat(params["FIXED_BATCH"], 1)
        probs_logits_ups = encoder_net.forward(x_ups)
        probs_logits_ups.retain_grad()
        if params["estimator_name"] == "REINFORCE":
            estimator_ups = REINFORCE(probs_logits_ups, target_ups, loss_func)
        estimator_ups.backward()
        # retrieve gradient of estimator [FIXED_BATCH, L]
        g_estimator = probs_logits_ups.grad
        for class_ind in range(num_classes):
            vars_grad[epoch, class_ind] = g_estimator.var(dim=0)[class_ind].item()
    results = {"expected_losses": expected_losses, "vars_grad": vars_grad.sum(1)}
    return results


if __name__ == "__main__":
    run_experiment()