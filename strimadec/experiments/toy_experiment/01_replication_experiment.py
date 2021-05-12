import pathlib
import os

import torch
import numpy as np

from strimadec.experiments.toy_experiment.utils import (
    run_stochastic_optimization,
    plot_toy,
    build_experimental_setup,
)


def run_experiment(train: bool, num_epochs=None, num_repetitions=None):
    """
        executes the replication experiment of the thesis, i.e., replicating
        the results of Grathwohl et al. (2018) using a Categorical distribution
        instead of a Bernoulli distribution

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    target = torch.tensor([0.499, 0.501]).unsqueeze(0)
    estimator_names = ["REINFORCE", "REBAR", "RELAX", "Exact gradient"]
    # estimator_names = ["REINFORCE", "RELAX"]
    # define path where to store/load results
    store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")
    if train:  # run experiment and save results in file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            losses = np.zeros([num_repetitions, num_epochs])
            vars_grad = np.zeros([num_repetitions, num_epochs])
            elapsed_times = np.zeros([num_repetitions, num_epochs])
            for i_experiment in range(num_repetitions):
                print(f"Start {estimator_name}-estimator {i_experiment + 1}/{num_repetitions} ...")
                SEED = i_experiment
                params = build_experimental_setup(estimator_name, target, num_epochs, SEED)
                current_results_dict = run_stochastic_optimization(params)

                losses[i_experiment] = current_results_dict["expected_losses"]
                vars_grad[i_experiment] = current_results_dict["vars_grad"]
                elapsed_times[i_experiment] = current_results_dict["elapsed_times"]
            results[estimator_name] = {
                "losses": losses,
                "vars_grad": vars_grad,
                "elapsed_times": elapsed_times,
            }
            # store current estimator results
            store_path_estimator = f"{store_dir}/replication_experiment_{estimator_name}.npy"
            np.save(store_path_estimator, results[estimator_name])
    else:  # load experimental results from files
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            store_path_estimator = f"{store_dir}/replication_experiment_{estimator_name}.npy"
            estimator_results = np.load(store_path_estimator, allow_pickle=True).item()
            results[estimator_name] = estimator_results
    # plot results and store them
    store_path_fig = f"{store_dir}/replication_experiment.pdf"
    plot_toy(results, store_path_fig)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        default=True,
        action="store_false",
        dest="train",
        help="load results instead of actual training",
    )
    parser.add_argument(
        "--num_epochs", default=None, action="store", type=int, help="number of epochs to train"
    )
    parser.add_argument(
        "--num_repetitions",
        default=None,
        action="store",
        type=int,
        help="number of repetitions for each estimator experiment",
    )
    parse_results = parser.parse_args()
    train = parse_results.train
    num_epochs = parse_results.num_epochs
    num_repetitions = parse_results.num_repetitions

    run_experiment(train=train, num_epochs=num_epochs, num_repetitions=num_repetitions)
