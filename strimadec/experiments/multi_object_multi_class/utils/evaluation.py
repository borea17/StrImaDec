import os 
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')

from strimadec.models import AIR, DAIR


def evaluation(store_dir, start_folders, model_names):
    results = {}
    for model_name in model_names:
        results[model_name] = {}
    for i_folder, start_folder in enumerate(start_folders):
        model_name = model_names[i_folder]
        experiment_folder = os.path.join(store_dir, start_folder)
        ckpt_paths = glob.glob(f"{experiment_folder}/**/*.ckpt", recursive=True)
        num_reps = len(ckpt_paths)
        
        NLLs, count_accs, REINFORCEs = [], [], []
        trained_models = []
        for i_check, checkpoint_path in enumerate(ckpt_paths):
            if model_name == "AIR":
                trained_model = AIR.load_from_checkpoint(checkpoint_path=checkpoint_path)
            elif model_name == "DAIR":
                trained_model = DAIR.load_from_checkpoint(checkpoint_path=checkpoint_path)
            trained_model.eval()
            trained_model.freeze()
            print(f"Computing Evaluation Metrics for {model_name} Model: {i_check+1}/{num_reps}")
            evaluation_dict = trained_model.compute_evaluation_metrics()
            NLLs.append(evaluation_dict["NLL"])
            count_accs.append(100*evaluation_dict["acc"])
            REINFORCEs.append(evaluation_dict["REINFORCE_term"])
            trained_models.append(trained_model)
        results[model_name] = {
            "best acc": np.round(max(count_accs), 2),
            "worst acc": np.round(min(count_accs), 2),
            "avg acc": np.round(np.mean(count_accs), 2),
            "best NLL": np.round(min(NLLs), 2),
            "worst NLL": np.round(max(NLLs), 2),
            "avg NLL": np.round(np.mean(NLLs), 2),
            "best REINFORCE": np.round(min(REINFORCEs), 2),
            "worst REINFORCE": np.round(max(REINFORCEs), 2),
            "avg REINFORCE": np.round(np.mean(REINFORCEs), 2)
        }
        # make plots
        best_acc_ind =  np.argmax(count_accs)
        trained_models[best_acc_ind].plot_reconstructions(os.path.join(experiment_folder, "recs.pdf"))
        if model_name == "DAIR":
            trained_models[best_acc_ind].plot_latents(os.path.join(experiment_folder, "best_acc.pdf"))
            best_NLL_ind = np.argmin(NLLs)
            trained_models[best_NLL_ind].plot_latents(os.path.join(experiment_folder, "best_NLLs.pdf"))
    return results
        