import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR


def run_experiment():
    estimator_names = ["analytical", "REINFORCE", "NVIL", "CONCRETE", "REBAR"]
    # estimator_names = ["analytical"]
    # define setup
    setup_dict = {
        # "target": torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0),
        "target": torch.tensor([0.45, 0.55]).unsqueeze(0),
        "loss_func": lambda x, y: (x - y) ** 2,
        "num_epochs": 5000,
    }
    # store results in results_dict
    results_dict = {}
    for i, estimator_name in enumerate(estimator_names):
        params = build_experimental_setup(estimator_name, setup_dict)
        results_dict[str(i)] = run_stochastic_optimization(params)
        results_dict[str(i)]["params"] = params
    plot(setup_dict, results_dict)
    return


def plot(setup_dict, results_dict):
    # compute optimal loss
    target = setup_dict["target"]
    loss_func = setup_dict["loss_func"]
    num_classes = target.shape[1]
    possible_losses = torch.zeros(num_classes)
    for class_ind in range(num_classes):
        one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0)
        possible_losses[class_ind] = loss_func(one_hot_class, target).mean()
    optimal_loss = min(possible_losses)
    # start plot
    fig = plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(results_dict)):
        losses = results_dict[str(i)]["expected_losses"]
        steps = np.arange(len(losses))
        name = results_dict[str(i)]["params"]["estimator_name"]
        plt.plot(steps, losses, label=name)
    plt.plot([steps[0], steps[-1]], [optimal_loss, optimal_loss], label="optimal", color="gray")
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(len(results_dict)):
        log_vars_grad = np.log(results_dict[str(i)]["vars_grad"] + 1e-12)
        steps = np.arange(len(log_vars_grad))
        name = results_dict[str(i)]["params"]["estimator_name"]
        plt.plot(steps, log_vars_grad, label=name)
    plt.xlabel("Steps")
    plt.ylabel("Log (Var (Gradient Estimator) )")
    plt.legend()
    plt.show()
    return


def build_experimental_setup(estimator_name, setup_dict):
    """
        creates the experimental setup params given the estimator_name

    Args:
        estimator_name (str): name of gradient estimator
        setup_dict (dict): dictionary containing the following setup variables
            target (torch tensor): defines the target in the stochastic optimization problem [1, L]
            loss_func (method): loss function using sampled class and a target
            num_epochs (int):

    Returns:
        params (dict): dictionary that can be used to execute
            `run_stochastic_optimization`
    """
    x = torch.ones([1, 1])
    # use the simplest possible network (can be easily adapted)
    encoder_net = nn.Sequential(nn.Linear(1, setup_dict["target"].shape[1]))

    params = {
        "SEED": 42,
        "x": x,
        "target": setup_dict["target"],
        "encoder_net": encoder_net,
        "num_epochs": setup_dict["num_epochs"],
        "batch_size": 1,
        "lr": 0.01,
        "loss_func": setup_dict["loss_func"],
        "estimator_name": estimator_name,
        "FIXED_BATCH": 1000,
    }

    if estimator_name == "NVIL":
        baseline_net = nn.Sequential(nn.Linear(1, 1))
        params["baseline_net"] = baseline_net
        params["tune_lr"] = 0.1
    elif estimator_name == "CONCRETE":
        params["temp"] = 1.0
    elif estimator_name == "REBAR":
        params["eta"] = torch.tensor([1.0], requires_grad=True)
        params["log_temp"] = torch.tensor([0.0], requires_grad=True)
        params["tune_lr"] = 0.01
    return params


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
    if estimator_name in ["analytical", "REINFORCE", "CONCRETE"]:
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
        if params["estimator_name"] == "analytical":
            estimator = analytical(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "REINFORCE":
            estimator = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "NVIL":
            baseline_vals_ups = baseline_net.forward(x).repeat(params["batch_size"], 1)
            estimator = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif params["estimator_name"] == "CONCRETE":
            estimator = CONCRETE(probs_logits_ups, target_ups, params["temp"], loss_func)
        elif params["estimator_name"] == "REBAR":
            temp, eta = params["log_temp"].exp(), params["eta"]
            estimator = REBAR(probs_logits_ups, target_ups, temp, eta, params["loss_func"])
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
        if params["estimator_name"] == "analytical":
            estimator_ups = analytical(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "REINFORCE":
            estimator_ups = REINFORCE(probs_logits_ups, target_ups, loss_func)
        elif params["estimator_name"] == "NVIL":
            baseline_vals_ups = baseline_net.forward(x_ups)
            estimator_ups = NVIL(probs_logits_ups, target_ups, baseline_vals_ups, loss_func)
        elif params["estimator_name"] == "CONCRETE":
            estimator_ups = CONCRETE(probs_logits_ups, target_ups, params["temp"], loss_func)
        elif params["estimator_name"] == "REBAR":
            temp, eta = params["log_temp"].exp(), params["eta"]
            estimator_ups = REBAR(probs_logits_ups, target_ups, temp, eta, params["loss_func"])
        estimator_ups.backward()
        # retrieve gradient of estimator [FIXED_BATCH, L]
        g_estimator = probs_logits_ups.grad
        for class_ind in range(num_classes):
            vars_grad[epoch, class_ind] = g_estimator.var(dim=0)[class_ind].item()
    results = {"expected_losses": expected_losses, "vars_grad": vars_grad.sum(1)}
    return results


if __name__ == "__main__":
    run_experiment()