import os
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')

from strimadec.models import DVAE, DVAEST

def overclustering_evaluation(store_dir, start_folders, estimator_names, model_names):
    """
        computes evaluation metrics,
        makes plots for worst and best scores for overclustering experiment
    """
    num_overclusters = len(start_folders)
    results = np.zeros(num_overclusters, dtype=object)
    for i_overclusters in range(3):
        results[i_overclusters] = dict()
    for i, estimator_name in enumerate(estimator_names):
        for i_overclusters, folder_start in enumerate(start_folders):
            print(f"Start with {i_overclusters + 1}/{len(start_folders)} overclustering folders")
            name = f"{folder_start}_{estimator_name}"
            experiment_folder = os.path.join(store_dir, name)
            # find all checkpoint_paths (training is configured such that only the last step is stored)
            ckpt_paths = glob.glob(f"{experiment_folder}/**/*.ckpt", recursive=True)
            num_reps = len(ckpt_paths)
            NLLs, KL_Divs, accs, empties = [], [], [], []
            trained_models = []
            for i_check, checkpoint_path in enumerate(ckpt_paths):
                trained_model = DVAEST.load_from_checkpoint(checkpoint_path=checkpoint_path)
                trained_model.eval()
                trained_model.freeze()
                print(f"..computing Evaluation Metrics for {estimator_name} {i_check+1}/{num_reps}")
                evaluation_dict = trained_model.compute_evaluation_metrics()
                # store results
                NLLs.append(evaluation_dict["NLL"])
                KL_Divs.append(evaluation_dict["KL_Div"])
                accs.append(100*evaluation_dict["acc"])
                empties.append(evaluation_dict["empty"])
                trained_models.append(trained_model)
            results[i_overclusters][estimator_name] = {
                "best acc": np.round(max(accs), 2),
                "worst acc": np.round(min(accs), 2),
                "avg acc": np.round(np.mean(accs), 2),
                "best NLL": np.round(min(NLLs), 2),
                "worst NLL": np.round(max(NLLs), 2),
                "avg NLL": np.round(np.mean(NLLs), 2),
                "best KL": np.round(min(KL_Divs), 2),
                "worst KL": np.round(max(KL_Divs), 2),
                "avg KL": np.round(np.mean(KL_Divs), 2),
            }
            # make plots
            best_index, worst_index = np.argmax(accs), np.argmin(accs) 
            store_path_best_fig = os.path.join(experiment_folder, "best.pdf")
            trained_models[best_index].plot_latents(store_path_best_fig)
            store_path_best_rec_fig = os.path.join(experiment_folder, "best_reconstruction.pdf")
            trained_models[best_index].plot_reconstructions(store_path_best_rec_fig)
            store_path_worst_fig = os.path.join(experiment_folder, "worst.pdf")
            trained_models[worst_index].plot_latents(store_path_worst_fig)
    return results


def standard_evaluation(store_dir, folder_start, estimator_names, model_name):
    """
        computes evaluation metrics,
        makes plots for worst and best score for normal clustering experiments
    """
    results = {}
    for i, estimator_name in enumerate(estimator_names):
        name = f"{folder_start}_{estimator_name}"
        experiment_folder = os.path.join(store_dir, name)
        # find all checkpoint_paths (training is configured such that only the last step is stored)
        ckpt_paths = glob.glob(f"{experiment_folder}/**/*.ckpt", recursive=True)
        num_reps = len(ckpt_paths)
        NLLs, KL_Divs, accs, empties = [], [], [], []
        trained_models = []
        for i_check, checkpoint_path in enumerate(ckpt_paths):
            if model_name == "D-VAE":
                trained_model = DVAE.load_from_checkpoint(checkpoint_path=checkpoint_path)
            elif model_name == "D-VAE-ST":
                trained_model = DVAEST.load_from_checkpoint(checkpoint_path=checkpoint_path)
            trained_model.eval()
            trained_model.freeze()
            print(f"Computing Evaluation Metrics for {estimator_name} {i_check+1}/{num_reps}")
            evaluation_dict = trained_model.compute_evaluation_metrics()
            # store results
            NLLs.append(evaluation_dict["NLL"])
            KL_Divs.append(evaluation_dict["KL_Div"])
            accs.append(100*evaluation_dict["acc"])
            empties.append(evaluation_dict["empty"])
            trained_models.append(trained_model)
        results[estimator_name] = {
            "best acc": np.round(max(accs), 2),
            "worst acc": np.round(min(accs), 2),
            "avg acc": np.round(np.mean(accs), 2),
            "best NLL": np.round(min(NLLs), 2),
            "worst NLL": np.round(max(NLLs), 2),
            "avg NLL": np.round(np.mean(NLLs), 2),
            "best KL": np.round(min(KL_Divs), 2),
            "worst KL": np.round(max(KL_Divs), 2),
            "avg KL": np.round(np.mean(KL_Divs), 2),
        }
        # make plots
        best_index, worst_index = np.argmax(accs), np.argmin(accs) 
        store_path_best_fig = os.path.join(experiment_folder, "best.pdf")
        trained_models[best_index].plot_latents(store_path_best_fig)
        store_path_best_rec_fig = os.path.join(experiment_folder, "best_reconstruction.pdf")
        trained_models[best_index].plot_reconstructions(store_path_best_rec_fig)
        store_path_worst_fig = os.path.join(experiment_folder, "worst.pdf")
        trained_models[worst_index].plot_latents(store_path_worst_fig)
    return results

def load_parse_results(experiment_id, dataset_name, DECODER_DIST, estimator_names, num_clusters):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    results = {}

    for estimator_name in estimator_names:
        experiment_name = f"{dataset_name}_{DECODER_DIST}_{estimator_name}"
        estimator_df = df[df["run"].str.split("/").str[-2] == experiment_name]

        accs = estimator_df[estimator_df["tag"] == "metrics/deterministic_accuracy"]
        NLLs = estimator_df[estimator_df["tag"] == "losses/NLL"]
        KL_divs = estimator_df[estimator_df["tag"] == "losses/KL-Div"]
        elaps_times = estimator_df[estimator_df["tag"] == "elapsed_times/time"]

        # convert into array [num_repetitions, num_epochs] (each estimator has same num_epochs)
        num_repetitions = len(estimator_df["run"].unique())
        num_epochs = int(estimator_df[estimator_df["tag"] == "epoch"].value.max() + 1)
        # accuracy is not logged every epoch but only every x epochs
        num_acc_epochs = len(accs.step.unique())

        accs_arr = np.zeros([num_repetitions, num_acc_epochs])
        NLLs_arr = np.zeros([num_repetitions, num_epochs])
        KL_divs_arr = np.zeros([num_repetitions, num_epochs])
        times_arr = np.zeros([num_repetitions, num_epochs])
        for i_repetition in range(num_repetitions):
            repetition_name = f"{experiment_name}/version_{i_repetition}"

            accs_arr[i_repetition, :] = accs[accs["run"] == repetition_name].value
            NLLs_arr[i_repetition, :] = NLLs[NLLs["run"] == repetition_name].value
            KL_divs_arr[i_repetition, :] = KL_divs[KL_divs["run"] == repetition_name].value
            times_arr[i_repetition, :] = elaps_times[elaps_times["run"] == repetition_name].value
        results[estimator_name] = {
            "accs": accs_arr,
            "NLLs": NLLs_arr,
            "KL_divs": KL_divs_arr,
            "elapsed_times": times_arr
        }
    return results
    