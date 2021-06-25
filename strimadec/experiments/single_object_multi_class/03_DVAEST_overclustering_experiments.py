import os
import pathlib
from strimadec.experiments.toy_experiment.utils import experimental_setup

import tensorboard as tb
import numpy as np
from prettytable import PrettyTable

from strimadec.experiments.single_object_multi_class.utils import (
    run_single_experiment, overclustering_evaluation
)

store_dir = pathlib.Path(__file__).resolve().parents[0]


def run_experiment(train: bool, dataset_name: str, num_epochs, num_repetitions):
    """
        executes the D-VAE-ST robustness experiment of the thesis, i.e., using more clusters than
        classes for the FullMNIST dataset

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        dataset_name (str): name of dataset ["SimplifiedMNIST", "FullMNIST", "Letters"]
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    MODEL_NAME, DECODER_DIST = "D-VAE-ST_Overclustering", "Gaussian"
    if dataset_name in ["FullMNIST", "FashionMNIST"]:
        num_clusters_list = [10, 11, 12]
    elif dataset_name == "SimplifiedMNIST":
        num_clusters_list = [3, 4, 5]
    if train:  # results can be visualized via `tensorboard --logdir DVAE_results`
        for i, estimator_name in enumerate(estimator_names):
            for i_experiment in range(num_repetitions):
                print(f"Start {estimator_name}-estimator {i_experiment + 1}/{num_repetitions} ...")
                SEED = i_experiment
                for num_clusters in num_clusters_list:
                    # run single experiment (results are automatically stored by tensorboard)
                    run_single_experiment(
                        model_name=MODEL_NAME,
                        estimator_name=estimator_name,
                        num_epochs=num_epochs,
                        dataset_name=dataset_name,
                        decoder_dist=DECODER_DIST,
                        num_clusters=num_clusters,
                        store_dir=store_dir,
                        SEED=SEED,
                    )
    # load models, compute evaluation metrics and make plots
    start_folders = []
    for num_clusters in num_clusters_list:
        start_folders.append(f"{MODEL_NAME}_results/{dataset_name}_{num_clusters}_{DECODER_DIST}")
    results = overclustering_evaluation(store_dir, start_folders, estimator_names, MODEL_NAME)
    # create table
    table = PrettyTable(
        ["gradient estimator"] + 
        ["best acc 0", "worst acc 0", "avg acc 0"] +
        ["best acc 1", "worst acc 1", "avg acc 1"] +
        ["best acc 2", "worst acc 2", "avg acc 2"]
    )
    for estimator_name in estimator_names:
        row = [estimator_name]
        for over_clusters in range(len(num_clusters_list)):
            cur_results = results[over_clusters][estimator_name]
            row.extend([cur_results["best acc"], cur_results["worst acc"], cur_results["avg acc"]])
        table.add_row(row) 
    print(f"{dataset_name} {MODEL_NAME} Experiment")
    print(table)
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
        "--num_epochs", default=150, action="store", type=int, help="number of epochs to train"
    )
    parser.add_argument(
        "--num_repetitions",
        default=1,
        action="store",
        type=int,
        help="number of repetitions experiment shall be performed",
    )
    parser.add_argument(
        "--dataset_name",
        default="SimplifiedMNIST",
        choices=["SimplifiedMNIST", "FullMNIST", "Letters", "FashionMNIST"],
        action="store",
        type=str,
        help="name of dataset",
    )
    parse_results = parser.parse_args()

    train = parse_results.train
    num_epochs = parse_results.num_epochs
    num_repetitions = parse_results.num_repetitions
    dataset_name = parse_results.dataset_name

    run_experiment(train, dataset_name, num_epochs, num_repetitions)

