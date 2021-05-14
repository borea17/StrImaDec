import pathlib
import os
from numpy.core.fromnumeric import var

import torch
import numpy as np
from prettytable import PrettyTable

from strimadec.experiments.toy_experiment.utils import (
    run_hyperparameter_procedure,
    compute_optimal_loss,
    loss_func,
)

# fix target
target = torch.tensor([0.34, 0.33, 0.33]).unsqueeze(0)
# define path where to store/load results
store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "results")


def batch_size_experiment(train: bool, num_epochs=None, num_repetitions=None):
    """
        this experiment should highlight the significance of the batch-size for
        each gradient estimator


    Args:
        train (bool): decides whether experiment is executed or stored results are used
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    hyperparam_name = "batch_size"
    hyperparams = [1, 50]
    if train:  # run experiment and save results in file
        results = run_hyperparameter_procedure(
            estimator_names=estimator_names,
            hyperparam_name=hyperparam_name,
            hyperparams=hyperparams,
            num_epochs=num_epochs,
            num_repetitions=num_repetitions,
            target=target,
            store_dir=store_dir,
        )
    else:  # load experimental results from file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            store_path = f"{store_dir}/hyperparameter_{hyperparam_name}_exp_{estimator_name}.npy"
            estimator_results = np.load(store_path, allow_pickle=True).item()
            results[estimator_name] = estimator_results
    # make a table summary
    column_names = ["gradient estimator"]
    for hyperparam in hyperparams:
        column_names.extend(
            [
                f"{hyperparam} avg abs error [1000]",
                f"{hyperparam} avg log var [1]",
                f"{hyperparam} avg step time [ms]",
            ]
        )
    table = PrettyTable(column_names)
    optimal_loss = compute_optimal_loss(target, loss_func).item()
    for name_of_estimator, data_est in results.items():
        column_inp = [name_of_estimator]
        for i, hyperparam in enumerate(hyperparams):
            avg_error = (
                (np.abs(data_est[f"losses_{hyperparam_name}_{i}"] - optimal_loss)).mean(0).mean(0)
            )
            avg_log_vars = np.log(data_est[f"vars_grad_{hyperparam_name}_{i}"]).mean(0).mean(0)
            avg_step = data_est[f"elapsed_times_{hyperparam_name}_{i}"].mean()
            column_inp.extend(
                [
                    np.round(1000 * avg_error, 2),
                    np.round(avg_log_vars, 2),
                    np.round(1000 * avg_step, 2),
                ]
            )
        table.add_row(column_inp)
    print("Batch-size Experiment")
    print(table)
    # plot results and store them
    store_path_fig = f"{store_dir}/hyperparameter_{hyperparam_name}_exp.pdf"
    return


def tune_lr_experiment(train: bool, num_epochs=None, num_repetitions=None):
    """
        this experiment should highlight the significance of the tunining learning
        for the gradient estimators in which the variance is decreased via an additional
        tuneable parameter

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    estimator_names = ["NVIL", "REBAR", "RELAX", "Exact gradient"]
    hyperparam_name = "tune_lr"
    hyperparams = [0.1, 0.01, 0.001]
    if train:  # run experiment and save results in file
        results = run_hyperparameter_procedure(
            estimator_names=estimator_names,
            hyperparam_name=hyperparam_name,
            hyperparams=hyperparams,
            num_epochs=num_epochs,
            num_repetitions=num_repetitions,
            target=target,
            store_dir=store_dir,
        )
    else:  # load experimental results from file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            store_path = f"{store_dir}/hyperparameter_{hyperparam_name}_exp_{estimator_name}.npy"
            estimator_results = np.load(store_path, allow_pickle=True).item()
            results[estimator_name] = estimator_results
    # make a table summary
    column_names = ["gradient estimator"]
    for hyperparam in hyperparams:
        column_names.extend(
            [
                f"{hyperparam} avg abs error [1000]",
                f"{hyperparam} avg log var [1]",
            ]
        )
    table = PrettyTable(column_names)
    optimal_loss = compute_optimal_loss(target, loss_func).item()
    for name_of_estimator, data_est in results.items():
        column_inp = [name_of_estimator]
        for i, hyperparam in enumerate(hyperparams):
            avg_error = (
                (np.abs(data_est[f"losses_{hyperparam_name}_{i}"] - optimal_loss)).mean(0).mean(0)
            )
            avg_log_vars = np.log(data_est[f"vars_grad_{hyperparam_name}_{i}"]).mean(0).mean(0)
            column_inp.extend(
                [
                    np.round(1000 * avg_error, 2),
                    np.round(avg_log_vars, 2),
                ]
            )
        table.add_row(column_inp)
    print("Tune Learning Rate Experiment")
    print(table)
    # plot results and store them
    store_path_fig = f"{store_dir}/hyperparameter_{hyperparam_name}_exp.pdf"
    return


def CONCRETE_experiment(train: bool, num_epochs=None, num_repetitions=None):
    """
        this experiment should highlight the significance of temperature parameter
        in the CONCRETE gradient estimator

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    hyperparam_name = "temp"
    hyperparams = [0.2, 0.6, 2.0]
    estimator_names = ["CONCRETE", "Exact gradient"]
    if train:  # run experiment and save results in file
        results = run_hyperparameter_procedure(
            estimator_names=estimator_names,
            hyperparam_name=hyperparam_name,
            hyperparams=hyperparams,
            num_epochs=num_epochs,
            num_repetitions=num_repetitions,
            target=target,
            store_dir=store_dir,
        )
    else:  # load experimental results from file
        results = {}
        for i, estimator_name in enumerate(estimator_names):
            store_path = f"{store_dir}/hyperparameter_{hyperparam_name}_exp_{estimator_name}.npy"
            estimator_results = np.load(store_path, allow_pickle=True).item()
            results[estimator_name] = estimator_results
    # make a table summary
    column_names = ["gradient estimator"]
    for hyperparam in hyperparams:
        column_names.extend(
            [
                f"{hyperparam} avg abs error [1000]",
                f"{hyperparam} avg log var [1]",
            ]
        )
    table = PrettyTable(column_names)
    optimal_loss = compute_optimal_loss(target, loss_func).item()
    for name_of_estimator, data_est in results.items():
        column_inp = [name_of_estimator]
        for i, hyperparam in enumerate(hyperparams):
            avg_error = (
                (np.abs(data_est[f"losses_{hyperparam_name}_{i}"] - optimal_loss)).mean(0).mean(0)
            )
            avg_log_vars = np.log(data_est[f"vars_grad_{hyperparam_name}_{i}"]).mean(0).mean(0)
            column_inp.extend(
                [
                    np.round(1000 * avg_error, 2),
                    np.round(avg_log_vars, 2),
                ]
            )
        table.add_row(column_inp)
    print("CONCRETE Temperature Experiment")
    print(table)
    # plot results and store them
    store_path_fig = f"{store_dir}/hyperparameter_{hyperparam_name}_exp.pdf"
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

    batch_size_experiment(train=train, num_epochs=num_epochs, num_repetitions=num_repetitions)
    tune_lr_experiment(train=train, num_epochs=num_epochs, num_repetitions=num_repetitions)
    CONCRETE_experiment(train=train, num_epochs=num_epochs, num_repetitions=num_repetitions)
