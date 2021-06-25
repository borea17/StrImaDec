import os
import pathlib

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from prettytable import PrettyTable

from strimadec.models import AIR, DAIR
from strimadec.experiments.multi_object_multi_class.utils import run_single_experiment, evaluation


store_dir = pathlib.Path(__file__).resolve().parents[0]


def run_experiment(train: bool, dataset_name: str, num_epochs, num_repetitions):
    """
        executes the multi-object-multi-class experiment of the thesis

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        dataset_name (str): name of dataset ["SimplifiedMultiMNIST", "MultiMNIST"]
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    if dataset_name == "SimplifiedMultiMNIST":
        num_clusters = 3
    elif dataset_name == "MultiMNIST":
        num_clusters = 10
    model_names = ["AIR", "DAIR"]
    if train:
        for model_name in model_names:
            for i_experiment in range(num_repetitions):
                print(f"Start {model_name} experiment {i_experiment + 1}/{num_repetitions}")
                SEED = i_experiment
                # run single experiment
                run_single_experiment(
                    model_name=model_name,
                    num_epochs=num_epochs,
                    dataset_name=dataset_name,
                    num_clusters=num_clusters, 
                    store_dir=store_dir,
                    SEED=SEED
                )
    # load models, compute evaluation metrics and make plots
    start_folders = []
    for model_ame in model_names:
        start_folders.append(f"Comparison/{model_ame}_results/NVIL_{num_clusters}_{dataset_name}")
    results = evaluation(store_dir, start_folders, model_names)
    # create table
    col_names = (
        ["model"] + 
        ["best acc", "worst acc", "avg acc"] +
        ["best NLL", "worst NLL", "avg NLL"] +
        ["best REINFORCE", "worst REINFORCE", "avg REINFORCE"]
    )
    table = PrettyTable(col_names)
    for model_name in model_names:
        row = [model_name]
        for col_name in col_names[1::]: 
            row.append(results[model_name][col_name])
        table.add_row(row) 
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
        default="SimplifiedMultiMNIST",
        choices=["SimplifiedMultiMNIST", "MultiMNIST"],
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