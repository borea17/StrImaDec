def build_DAIR_config(dataset_name, num_epochs, num_clusters, SEED):
    img_channels, img_dim = 1, 64
    window_size = 28
    # initialization of p_pres, mu_where and log_where
    p_pres_init = [1.0]  # (sigmoid -> 0.8)
    mu_where_init = [3.0, 0.0, 0.0]
    log_var_where_init = [-3.0, -3.0, -3.0]
    config = {
        "SEED": SEED,
        "dataset_name": dataset_name,
        "experiment_name": f"NVIL_{num_clusters}_{dataset_name}",
        "num_epochs": num_epochs,
        "lr": 1e-4,
        "weight_decay": 1e-7,
        "base_lr": 1e-2,
        "base_weight_decay": 1e-7,
        "img_shape": [img_channels, img_dim, img_dim],
        "log_every_k_epochs": 10,
        "number_of_slots_train": 3,
        "prior_z_pres": 0.01,
        "prior_mean_z_where": [3.0, 0.0, 0.0],
        "prior_var_z_where": [0.1 ** 2, 1.0, 1.0],
        "What VAE-Setup": {
            "img_channels": img_channels,
            "img_dim": window_size,
            "FC_hidden_dims_enc": [400, 400],
            "FC_hidden_dims_dec": [800],
            "latent_dim": num_clusters,
            "encoder_distribution": "Categorical",
            "decoder_distribution": "Gaussian",
            "fixed_var": 0.3,
            "enforce_positivity": True
        },
        "RNN-Setup": {
            "baseline_net": False,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + num_clusters,
            "FC_hidden_dims": [400],
            "output_size": 1 + 2 * 3,
            "output_bias_init": p_pres_init + mu_where_init + log_var_where_init,
        },
        "RNN Baseline-Setup": {
            "baseline_net": True,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + num_clusters,
            "FC_hidden_dims": [400],
            "output_size": 1,
            "output_bias_init": p_pres_init,
        },
        "Localization-Setup": {
            "input_dim": img_channels * window_size * window_size,
            "FC_hidden_dims": [400, 400],
            "prior_mu_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # identity transform
            "prior_var_transform": [0.01] * 6,
        },
    }
    return config


def build_AIR_config(dataset_name, num_epochs, num_clusters, SEED):
    img_channels, img_dim = 1, 64
    window_size = 28
    # initialization of p_pres, mu_where and log_where
    p_pres_init = [2.0]  # (sigmoid -> 0.8)
    mu_where_init = [3.0, 0.0, 0.0]
    log_var_where_init = [-3.0, -3.0, -3.0]
    config = {
        "SEED": SEED,
        "dataset_name": dataset_name,
        "experiment_name": f"NVIL_{num_clusters}_{dataset_name}",
        "num_epochs": num_epochs,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "base_lr": 1e-2,
        "base_weight_decay": 1e-6,
        "img_shape": [img_channels, img_dim, img_dim],
        "log_every_k_epochs": 1,
        "number_of_slots_train": 3,
        "prior_z_pres": 0.01,
        "prior_mean_z_where": [3.0, 0.0, 0.0],
        "prior_var_z_where": [0.1 ** 2, 1.0, 1.0],
        "What VAE-Setup": {
            "img_channels": img_channels,
            "img_dim": window_size,
            "FC_hidden_dims_enc": [400],
            "FC_hidden_dims_dec": [400],
            "latent_dim": num_clusters,
            "encoder_distribution": "Gaussian",
            "decoder_distribution": "Gaussian",
            "fixed_var": 0.2,
            "enforce_positivity": True
        },
        "RNN-Setup": {
            "baseline_net": False,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + num_clusters,
            "FC_hidden_dims": [400],
            "output_size": 1 + 2 * 3,
            "output_bias_init": p_pres_init + mu_where_init + log_var_where_init,
        },
        "RNN Baseline-Setup": {
            "baseline_net": True,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + num_clusters,
            "FC_hidden_dims": [256, 256],
            "output_size": 1,
            "output_bias_init": p_pres_init,
        },
    }
    return config
