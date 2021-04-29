import pathlib
import os

import torch
import numpy as np

from strimadec.experiments.toy_experiment.utils import run_stochastic_optimization, plot_replication


def run_experiment(run=True):
    """
        executes the replication experiment of the thesis, i.e., replicating
        the results of Grathwohl et al. (2018) using a Categorical distribution
        instead of a Bernoulli distribution

    Args:
        run (bool): decides whether experiment is executed or stored results are used
    """
    # estimator_names = ["REINFORCE", "REBAR", "RELAX", "Exact gradient"]
    estimator_names = ["NVIL tune_lr=0.1", "NVIL tune_lr=0.001"]
    batch_sizes = [1, 100]
    tune_lrs = [0.1, 0.001]

    # define path where to store/load results
    store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")
    store_path = f"{store_dir}/replication_experiment.npy"
    if run:  # run experiment and save results in file
        # define params
        results = []
        for i, estimator_name in enumerate(estimator_names):
            print(f"Start Experiment with {estimator_name}-estimator...")
            params = build_experimental_setup(estimator_name)
            params["tune_lr"] = tune_lrs[i]
            current_dict = run_stochastic_optimization(params)
            current_dict["name"] = estimator_name
            results.append(current_dict)
        np.save(store_path, results)
    else:  # load experimental results from file
        results = np.load(store_path, allow_pickle=True)
    store_path_fig = f"{store_dir}/replication_experiment.pdf"
    plot_replication(results, store_path_fig)
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
    target = torch.tensor([0.499, 0.501]).unsqueeze(0)
    SEED = 42  # 1
    torch.manual_seed(SEED)  # seed here to make network initializations deterministic
    # use the simplest possible network (can be easily adapted)
    num_classes = target.shape[1]
    encoder_net = torch.nn.Sequential(torch.nn.Linear(1, num_classes, bias=False))

    params = {
        "SEED": SEED,
        "x": x,
        "target": target,
        "encoder_net": encoder_net,
        "num_epochs": 5000,
        "batch_size": 1,
        "lr": 0.01,
        "loss_func": lambda x, y: (x - y) ** 2,
        "estimator_name": estimator_name,
        "FIXED_BATCH": 1000,
    }

    if "REBAR" in estimator_name:
        params["eta"] = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        params["log_temp"] = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        params["tune_lr"] = 0.001

    elif "NVIL" in estimator_name:
        baseline_net = torch.nn.Sequential(torch.nn.Linear(1, 1))
        params["baseline_net"] = baseline_net
        params["tune_lr"] = 0.1
    elif "RELAX" in estimator_name:

        class C_PHI(torch.nn.Module):
            """
                Control variate for RELAX,
                NOTE: this is only the neural part of the control variate which will be used as
                      the adapted input in the loss function

            Args:
                num_classes (int): number of classes
                log_temp_init (float): logarithmized init temperature for continuous relaxation
            """

            def __init__(self, num_classes, log_temp_init):
                super(C_PHI, self).__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(num_classes, num_classes),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_classes, num_classes),
                )
                self.log_temp = torch.nn.Parameter(torch.tensor(log_temp_init), requires_grad=True)
                return

            def forward(self, z):
                temp = self.log_temp.exp()
                z_tilde = torch.softmax(z / temp, dim=1)
                out = self.network(z_tilde)
                return out

        params["tune_lr"] = 0.01
        params["c_phi"] = C_PHI(num_classes=num_classes, log_temp_init=0.0)
    return params


if __name__ == "__main__":
    run_experiment(run=True)
