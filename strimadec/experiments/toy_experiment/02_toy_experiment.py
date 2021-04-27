import os
import pathlib

import torch
import numpy as np

from strimadec.experiments.toy_experiment import run_stochastic_optimization, plot_toy_results


def run_experiment(run=True):
    """
        executes the toy experiment of the thesis, i.e., a 3-class categorical distribution is
        used to compare the different discrete gradient estimators

    Args:
        run (bool): decides whether experiment is executed or stored results are used
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "analytical"]
    # define path where to store/load results
    store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")
    store_path = f"{store_dir}/toy_experiment.npy"
    print(store_path)
    if run:  # run experiment and save results in file
        # define params
        target = torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0)
        results = []
        for i, estimator_name in enumerate(estimator_names):
            print(f"Start Experiment with {estimator_name}-estimator...")
            params = build_experimental_setup(estimator_name)
            current_dict = run_stochastic_optimization(params)
            current_dict["name"] = estimator_name
            results.append(current_dict)
        np.save(store_path, results)
    else:  # load experimental results from file
        results = np.load(store_path, allow_pickle=True)
    plot_toy_results(results)
    return