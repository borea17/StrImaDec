import pathlib
import os

import torch
import numpy as np

from strimadec.experiments.toy_experiment.utils import (
    run_stochastic_optimization,
    build_experimental_setup,
)


def batch_size_experiment(train: bool, num_epochs=None, num_repetitions=None):
    """
        this experiment should highlight the significance of the batch-size for
        each gradient estimator


    Args:
        train (bool): decides whether experiment is executed or stored results are used
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    target = torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0)
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    batch_sizes = [1, 50]
    # define path where to store/load results
    store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")
    if train:  # run experiment and save results in file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            losses_batch0 = np.zeros([num_repetitions, num_epochs])
            vars_grad_batch0 = np.zeros([num_repetitions, num_epochs])
            elapsed_times_batch0 = np.zeros([num_repetitions, num_epochs])
            losses_batch1 = np.zeros([num_repetitions, num_epochs])
            vars_grad_batch1 = np.zeros([num_repetitions, num_epochs])
            elapsed_times_batch1 = np.zeros([num_repetitions, num_epochs])
            for i_experiment in range(num_repetitions):
                print(f"Start {estimator_name}-estimator {i_experiment + 1}/{num_repetitions} ...")
                SEED = i_experiment
                params = build_experimental_setup(estimator_name, target, num_epochs, SEED)
                params["batch_size"] = batch_sizes[0]
                current_results_dict = run_stochastic_optimization(params)
                losses_batch0[i_experiment] = current_results_dict["expected_losses"]
                vars_grad_batch0[i_experiment] = current_results_dict["vars_grad"]
                elapsed_times_batch0[i_experiment] = current_results_dict["elapsed_times"]
                params["batch_size"] = batch_sizes[1]
                current_results_dict = run_stochastic_optimization(params)
                losses_batch1[i_experiment] = current_results_dict["expected_losses"]
                vars_grad_batch1[i_experiment] = current_results_dict["vars_grad"]
                elapsed_times_batch1[i_experiment] = current_results_dict["elapsed_times"]
            results[estimator_name] = {
                "losses_batch0": losses_batch0,
                "vars_grad_batch0": vars_grad_batch0,
                "elapsed_times_batch0": elapsed_times_batch0,
                "losses_batch1": losses_batch1,
                "vars_grad_batch1": vars_grad_batch1,
                "elapsed_times_batch1": elapsed_times_batch1,
            }
            # store current estimator results
            store_path = f"{store_dir}/hyperparameter_batch_experiment_{estimator_name}.npy"
            np.save(store_path, results[estimator_name])
    else:  # load experimental results from file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            store_path = f"{store_dir}/hyperparameter_batch_experiment_{estimator_name}.npy"
            estimator_results = np.load(store_path, allow_pickle=True).item()
            results[estimator_name] = estimator_results
    # plot results and store them
    pass


def tune_lr_experiment():
    pass


def CONCRETE_experiment():
    pass


if __name__ == "__main__":
    pass
