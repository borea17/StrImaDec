def experimental_setup(dataset_name, num_epochs, estimator_name, decoder_dist, num_clusters, SEED):
    img_channels, img_dim = 1, 28
    config = {
        "experiment_name": f"{dataset_name}_{num_clusters}_{decoder_dist}_{estimator_name}",
        "dataset_name": dataset_name,
        "VAE-Setup": {
            "encoder_distribution": "Categorical",
            "decoder_distribution": decoder_dist,
            "fixed_var": 0.15,
            "latent_dim": num_clusters,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "FC_hidden_dims_enc": [400, 400],
            "FC_hidden_dims_dec": [800],
        },
        "estimator_name": estimator_name,
        "lambda_1": 2 / 3,
        "lambda_2": 1.0,
        "lr": 5e-4,
        "weight_decay":  1e-6,
        "SEED": SEED,
        "num_epochs": num_epochs,
        "log_every_k_epochs": 10,
    }

    if estimator_name == "NVIL":
        config["Baseline-Setup"] = {
            "input_dim": img_channels * img_dim * img_dim,
            "FC_hidden_dims": [400, 400],
            "output_dim": 1,
        }
        config["tune_lr"] = 1e-3
        config["tune_weight_decay"] = 1e-6
    elif estimator_name == "REBAR":
        config["tune_lr"] = 1e-4
        config["tune_weight_decay"] = 1e-6
        config["eta"] = 1.0
        config["log_temp_init"] = 0.0
    elif estimator_name == "RELAX":
        config["tune_lr"] = 1e-4
        config["tune_weight_decay"] = 1e-6
        config["C_PHI-Setup"] = {
            "input_dim": num_clusters,
            "output_dim": num_clusters,
            "FC_hidden_dims": [400, 400],
            "log_temp_init": 0.0,
        }
    return config


def localization_setup():
    img_channels, img_dim = 1, 28
    localization_setup = {
        "input_dim": img_channels * img_dim * img_dim,
        "FC_hidden_dims": [400, 400],
        "prior_mu_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # identity transform
        "prior_var_transform": [0.01] * 6,
    }
    return localization_setup
