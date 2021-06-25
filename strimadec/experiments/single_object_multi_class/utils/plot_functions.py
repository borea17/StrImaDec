from strimadec.models import DVAE, DVAEST

def plot_best_and_worst_latents(store_dir, folder_start, estimator_names, model_name):
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
            

    pass

def plot_reconstructions(dataset_name):
    pass