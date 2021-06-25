import os
import glob
import pathlib
from strimadec.experiments.toy_experiment.utils import experimental_setup

import tensorboard as tb
import numpy as np
from prettytable import PrettyTable

from strimadec.experiments.single_object_multi_class.utils import (
    run_single_experiment, standard_evaluation
)
from strimadec.models import DVAE

store_dir = pathlib.Path(__file__).resolve().parents[0]

def run_experiment(train: bool, dataset_name: str, num_epochs, num_repetitions):
    """
        executes the D-VAE experiment of the thesis, i.e., testing several datasets
        using the D-VAE model

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        dataset_name (str): name of dataset ["SimplifiedMNIST", "FullMNIST", "Letters"]
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    if dataset_name == "SimplifiedMNIST":
        num_clusters = 3
    elif dataset_name in ["FullMNIST", "FashionMNIST"]:
        num_clusters = 10
    elif dataset_name == "Letters":
        num_clusters = 26
    MODEL_NAME, DECODER_DIST = "D-VAE", "Gaussian"
    if train: 
        for i, estimator_name in enumerate(estimator_names):
            for i_experiment in range(num_repetitions):
                print(f"Start {estimator_name}-estimator {i_experiment + 1}/{num_repetitions} ...")
                SEED = i_experiment


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
    folder_start = f"{MODEL_NAME}_results/{dataset_name}_{num_clusters}_{DECODER_DIST}"
    results = standard_evaluation(store_dir, folder_start, estimator_names, MODEL_NAME)
    # create table
    table = PrettyTable(
        ["gradient estimator"] + 
        ["best acc", "worst acc", "avg acc"] +
        ["best NLL", "worst NLL", "avg NLL"] +
        ["best KL", "worst KL", "avg KL"]
    )
    for estimator_name in estimator_names:
        cur_results = results[estimator_name]
        accs = [cur_results["best acc"], cur_results["worst acc"], cur_results["avg acc"]]
        NLLs = [cur_results["best NLL"], cur_results["worst NLL"], cur_results["avg NLL"]]
        KLs = [cur_results["best KL"], cur_results["worst KL"], cur_results["avg KL"]]
        table.add_row([estimator_name] + accs + NLLs + KLs) 
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
