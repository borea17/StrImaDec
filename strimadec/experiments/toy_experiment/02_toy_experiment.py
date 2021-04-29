import os
import pathlib

import torch
import numpy as np

from strimadec.experiments.toy_experiment.utils import run_stochastic_optimization, plot_toy


def run_experiment(run: bool, num_epochs: int, num_iterations: int):
    """
        executes the toy experiment of the thesis, i.e., a 3-class categorical distribution is
        used to compare the different discrete gradient estimators

    Args:
        run (bool): decides whether experiment is executed or stored results are used
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    estimator_names = ["REBAR"]
    # define path where to store/load results
    store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")
    store_path = f"{store_dir}/toy_experiment.npy"
    if run:  # run experiment and save results in file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            losses = np.zeros([num_iterations, num_epochs])
            vars_grad = np.zeros([num_iterations, num_epochs])
            elapsed_times = np.zeros([num_iterations, num_epochs])
            for i_experiment in range(num_iterations):
                print(f"Start {estimator_name}-estimator {i_experiment + 1}/{num_iterations} ...")
                SEED = i_experiment
                params = build_experimental_setup(estimator_name, num_epochs, SEED)
                current_results_dict = run_stochastic_optimization(params)
                vars_grad[i_experiment] = current_results_dict["expected_losses"]
                losses[i_experiment] = current_results_dict["vars_grad"]
                elapsed_times[i_experiment] = current_results_dict["elapsed_times"]
            results[estimator_name] = {
                "losses": losses,
                "vars_grad": vars_grad,
                "elapsed_times": elapsed_times,
            }
        np.save(store_path, results)
    else:  # load experimental results from file
        results = np.load(store_path, allow_pickle=True).item()
    print(type(results))
    store_path_fig = f"{store_dir}/toy_experiment.pdf"
    plot_toy(results, store_path_fig)
    return


def build_experimental_setup(estimator_name, num_epochs, SEED):
    """
        creates the experimental setup params given the estimator_name

    Args:
        estimator_name (str): name of gradient estimator
        num_epochs (int): number of epochs to train
        SEED (int): seed that shall be used for this experiment

    Returns:
        params (dict): dictionary that can be used to execute
            `run_stochastic_optimization`
    """
    x = torch.ones([1, 1])
    target = torch.tensor([0.499, 0.501]).unsqueeze(0)
    torch.manual_seed(SEED)  # seed here to make network initializations deterministic
    # use the simplest possible network (can be easily adapted)
    num_classes = target.shape[1]
    encoder_net = torch.nn.Sequential(torch.nn.Linear(1, num_classes, bias=False))

    params = {
        "SEED": SEED,
        "x": x,
        "target": target,
        "encoder_net": encoder_net,
        "num_epochs": num_epochs,
        "batch_size": 1,
        "lr": 0.01,
        "loss_func": lambda x, y: (x - y) ** 2,
        "estimator_name": estimator_name,
        "FIXED_BATCH": 1000,
    }

    if estimator_name == "NVIL":
        baseline_net = torch.nn.Sequential(torch.nn.Linear(1, 1))
        params["baseline_net"] = baseline_net
        params["tune_lr"] = 0.1
    elif estimator_name == "CONCRETE":
        params["temp"] = torch.tensor([0.6], requires_grad=False)
    elif estimator_name == "REBAR":
        params["eta"] = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        params["log_temp"] = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        params["tune_lr"] = 0.001
    elif estimator_name == "RELAX":

        class C_PHI(torch.nn.Module):
            """Control variate for RELAX"""

            def __init__(self, num_classes, log_temp_init):
                super(C_PHI, self).__init__()
                self.network = torch.nn.Sequential(torch.nn.Linear(num_classes, num_classes))
                self.log_temp = torch.nn.Parameter(torch.tensor(log_temp_init), requires_grad=True)
                return

            def forward(self, z):
                temp = self.log_temp.exp()
                z_tilde = torch.softmax(z / temp, dim=1)
                # return z_tilde
                out = self.network(z_tilde)
                return out

        params["tune_lr"] = 0.01
        params["c_phi"] = C_PHI(num_classes=num_classes, log_temp_init=0.5)
    return params


if __name__ == "__main__":
    run_experiment(run=True, num_epochs=5000, num_iterations=50)
