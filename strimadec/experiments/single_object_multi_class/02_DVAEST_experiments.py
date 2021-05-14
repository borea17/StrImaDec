import os
import pathlib
from strimadec.experiments.toy_experiment.utils import experimental_setup

import tensorboard as tb
import numpy as np

from strimadec.experiments.single_object_multi_class.utils import run_single_experiment


def run_experiment(train: bool, dataset_name: str, num_clusters, num_epochs, num_repetitions):
    """
        executes the D-VAE-ST experiment of the thesis, i.e., testing several datasets
        using the D-VAE-ST model

    Args:
        train (bool): decides whether experiment is executed or stored results are used
        dataset_name (str): name of dataset ["SimplifiedMNIST", "FullMNIST", "Letters"]
        num_clusters (int): latent dimension of DVAE
        num_epochs (int): number of epochs to train each estimator
        num_repetitions (int): number of repetitions for each estimator experiment
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    estimator_names = ["NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    MODEL_NAME, DECODER_DIST = "D-VAE-ST", "Gaussian"
    if train:  # results can be visualized via `tensorboard --logdir DVAE_results`
        for i, estimator_name in enumerate(estimator_names):
            store_dir = pathlib.Path(__file__).resolve().parents[0]

            ############################# TEMPORARY ##############################
            # delete saved results
            import shutil

            experiment_name = f"{MODEL_NAME}_results/{dataset_name}_{DECODER_DIST}_{estimator_name}"
            if os.path.exists(os.path.join(store_dir, experiment_name)):  # delete saved results
                shutil.rmtree(os.path.join(store_dir, experiment_name))
            ######################################################################

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
    else:  # load and parse results from https://tensorboard.dev/experiment/WZ3OLy1LRemRXNkuECYY8A/
        experiment_id = "WZ3OLy1LRemRXNkuECYY8A"
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        df = experiment.get_scalars()
        for estimator_name in estimator_names:
            experiment_name = f"{dataset_name}_{DECODER_DIST}_{estimator_name}"
            estimator_df = df[df["run"].str.split("/").str[-2] == experiment_name]

            accs = estimator_df[estimator_df["tag"] == "metrics/deterministic_accuracy"]
            NLLs = estimator_df[estimator_df["tag"] == "losses/NLL"]
            KL_divs = estimator_df[estimator_df["tag"] == "losses/KL-Div"]

            # convert into array [num_repetitions, num_epochs] (each estimator has same num_epochs)
            num_repetitions = len(estimator_df["run"].unique())
            num_epochs = int(estimator_df[estimator_df["tag"] == "epoch"].value.max() + 1)

            accs_arr = np.zeros([num_repetitions, num_epochs])
            NLLs_arr = np.zeros([num_repetitions, num_epochs])
            KL_divs_arr = np.zeros([num_repetitions, num_epochs])
            for i_repetition in range(num_repetitions):
                repetition_name = f"{experiment_name}/version_{i_repetition}"

                accs_arr[i_repetition, :] = accs[accs["run"] == repetition_name].value
                NLLs_arr[i_repetition, :] = NLLs[NLLs["run"] == repetition_name].value
                KL_divs_arr[i_repetition, :] = KL_divs[KL_divs["run"] == repetition_name].value


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
        choices=["SimplifiedMNIST", "FullMNIST", "Letters"],
        action="store",
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--num_clusters", default=3, action="store", type=int, help="latent dimension of VAE"
    )

    parse_results = parser.parse_args()

    train = parse_results.train
    num_epochs = parse_results.num_epochs
    num_repetitions = parse_results.num_repetitions
    dataset_name = parse_results.dataset_name
    num_clusters = parse_results.num_clusters

    run_experiment(train, dataset_name, num_clusters, num_epochs, num_repetitions)
