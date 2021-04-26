import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import uniform_filter1d

from strimadec.experiments.toy_experiment import run_stochastic_optimization


def run_experiment(run=True):
    # estimator_names = ["analytical", "REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX"]

    estimator_names = ["REINFORCE", "REBAR", "RELAX", "analytical"]
    # define setup
    setup_dict = {
        # "target": torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0),
        "target": torch.tensor([0.499, 0.501]).unsqueeze(0),
        "loss_func": lambda x, y: (x - y) ** 2,
        "num_epochs": 10000,
    }
    if run:  # run experiment and save results in file
        # save results in results_dict
        results = []
        for i, estimator_name in enumerate(estimator_names):
            print(f"Start Experiment with {estimator_name}-estimator...")
            params = build_experimental_setup(estimator_name, setup_dict)
            current_dict = run_stochastic_optimization(params)
            current_dict["name"] = params["estimator_name"]
            results.append(current_dict)
        # store results
        np.save("results.npy", results)
    else:  # load experimental results from file
        results = np.load("results.npy", allow_pickle=True)
        # import pdb

        # pdb.set_trace()
    # plot results
    plot(setup_dict, results)
    return


def plot(setup_dict, results):
    # compute optimal loss
    target = setup_dict["target"]
    loss_func = setup_dict["loss_func"]
    num_classes = target.shape[1]
    possible_losses = torch.zeros(num_classes)
    for class_ind in range(num_classes):
        one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0)
        possible_losses[class_ind] = loss_func(one_hot_class, target).mean()
    optimal_loss = min(possible_losses)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    colors = ["#1F77B4", "#2CA02C", "#D62728", "gray"]
    # start plot
    fig = plt.figure(figsize=(13, 5))
    plt.suptitle(f"target = {target}")
    plt.subplot(1, 2, 1)
    for i, results_dict in enumerate(results):
        losses = results_dict["expected_losses"]
        steps = np.arange(len(losses))
        name = results_dict["name"]
        plt.plot(steps, losses, label=name, color=colors[i])
    # plt.plot([steps[0], steps[-1]], [optimal_loss, optimal_loss], label="optimal", color="gray")
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, results_dict in enumerate(results):
        log_vars_grad = np.log(results_dict["vars_grad"] + 1e-12)
        steps = np.arange(len(log_vars_grad))
        name = results_dict["name"]
        if name != "analytical":
            # smoothen log_var
            smooth_log_var = uniform_filter1d(log_vars_grad, size=5, mode="reflect")
            plt.plot(steps, log_vars_grad, label=name, color=colors[i])
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
    num_classes = setup_dict["target"].shape[1]
    linear_layer_encoder = nn.Linear(1, num_classes, bias=False)
    linear_layer_encoder.weight.data.fill_(0.0)
    encoder_net = nn.Sequential(linear_layer_encoder)

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
    elif estimator_name == "RELAX":
        params["tune_lr"] = 0.001
        params["c_phi"] = C_PHI(num_classes=num_classes, log_temp_init=0.5)
    return params


class C_PHI(nn.Module):
    """Control variate for RELAX"""

    def __init__(self, num_classes, log_temp_init):
        super(C_PHI, self).__init__()
        self.network = nn.Sequential(nn.Linear(num_classes, 1))
        self.log_temp = nn.Parameter(torch.tensor(log_temp_init), requires_grad=True)
        return

    def forward(self, z):
        temp = self.log_temp.exp()
        z_tilde = F.softmax(z / temp, dim=1)
        out = self.network(z_tilde)
        return out


if __name__ == "__main__":
    run_experiment()