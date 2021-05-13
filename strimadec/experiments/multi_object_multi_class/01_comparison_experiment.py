import os
import pathlib

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from strimadec.models import AIR


def run_experiment():
    # TEMPORARY
    dataset_name = "MultiMNIST"
    num_epochs = 150
    SEED = 1

    config = build_AIR_config(dataset_name, num_epochs, SEED)
    # make experiment reproducible
    seed_everything(config["SEED"])
    # instantiate model
    model = AIR(config)
    # define logger
    file_dir = pathlib.Path(__file__).resolve().parents[0]
    store_dir = os.path.join(file_dir, f"AIR_results")
    logger = TensorBoardLogger(store_dir, name=config["dataset_name"])
    # define callback of model checkpoint
    checkpoint_callback = ModelCheckpoint(period=1)
    # initialize pytorch lightning trainer
    trainer = pl.Trainer(
        deterministic=True,
        gpus=1,
        track_grad_norm=2,
        gradient_clip_val=0,  # don't clip
        max_epochs=config["num_epochs"],
        progress_bar_refresh_rate=20,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    # train model
    trainer.fit(model)
    return


def build_AIR_config(dataset_name, num_epochs, SEED):
    img_channels, img_dim = 1, 64
    window_size = 28
    what_latent_dim = 20
    # initialization of p_pres, mu_where and log_where
    p_pres_init = [2.0]  # (sigmoid -> 0.8)
    mu_where_init = [3.0, 0.0, 0.0]
    log_var_where_init = [-3.0, -3.0, -3.0]
    config = {
        "SEED": SEED,
        "dataset_name": dataset_name,
        "num_epochs": num_epochs,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "base_lr": 1e-2,
        "base_weight_decay": 1e-6,
        "img_shape": [img_channels, img_dim, img_dim],
        "log_every_k_epochs": 1,
        "number_of_slots_train": 3,
        "prior_z_pres": [0.01],
        "prior_mean_z_where": [3.0, 0.0, 0.0],
        "prior_var_z_where": [0.1 ** 2, 1.0, 1.0],
        "What VAE-Setup": {
            "img_channels": img_channels,
            "img_dim": window_size,
            "FC_hidden_dims_enc": [400],
            "FC_hidden_dims_dec": [400],
            "latent_dim": what_latent_dim,
            "encoder_distribution": "Gaussian",
            "decoder_distribution": "Gaussian",
            "fixed_var": 0.2,
        },
        "RNN-Setup": {
            "baseline_net": False,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + what_latent_dim,
            # "FC_hidden_dims": [256, 256],
            "FC_hidden_dims": [400],
            "output_size": 1 + 2 * 3,
            "output_bias_init": p_pres_init + mu_where_init + log_var_where_init,
        },
        "RNN Baseline-Setup": {
            "baseline_net": True,
            "img_channels": img_channels,
            "img_dim": img_dim,
            "hidden_state_dim": 256,
            "latent_space_dim": 1 + 3 + what_latent_dim,
            "FC_hidden_dims": [256, 256],
            "output_size": 1,
            "output_bias_init": p_pres_init,
        },
    }
    return config


if __name__ == "__main__":
    run_experiment()