import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from strimadec.experiments.toy_experiment import run_stochastic_optimization, plot_toy_results


def run_experiment(run=True):
    estimator_names = ["REINFORCE", "REBAR", "RELAX", "analytical"]
    # define setup
    setup_dict = {
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
    # plot results
    plot_toy_results(setup_dict, results)
    return