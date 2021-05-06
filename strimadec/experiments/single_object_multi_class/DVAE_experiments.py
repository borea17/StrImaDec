import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from strimadec.models import DVAE


def run_experiment():
    dataset_name = "MNIST"
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"]
    estimator_name = "REINFORCE"
    estimator_name = "REBAR"
    # estimator_name = "RELAX"
    decoder_distribution = "Gaussian"
    config = build_config(dataset_name, estimator_name, decoder_distribution)
    # make experiment reproducible
    seed_everything(config["SEED"])
    # instantiate model
    model = DVAE(config)
    # define logger
    logger = TensorBoardLogger("DVAE", name=config["experiment_name"])
    # define callback of model checkpoint (save model checkpoints and keep them)
    checkpoint_callback = ModelCheckpoint(period=1, save_top_k=-1)
    callbacks = [checkpoint_callback]
    # initialize pytorch lightning trainer
    trainer = pl.Trainer(
        deterministic=True,
        gpus=1,
        track_grad_norm=2,
        # gradient_clip_val=2.5,
        max_epochs=config["num_epochs"],
        progress_bar_refresh_rate=20,
        logger=logger,
        callbacks=callbacks,
    )
    # train model
    trainer.fit(model)
    return


def build_config(dataset_name, estimator_name, decoder_distribution):
    config = {
        "experiment_name": f"{dataset_name}_{decoder_distribution}_{estimator_name}",
        "dataset_name": dataset_name,
        "VAE-Setup": {
            "encoder_distribution": "Categorical",
            "decoder_distribution": decoder_distribution,
            "fixed_var": 0.3,
            "latent_dim": 3,
            "img_channels": 1,
            "img_dim": 28,
            "FC_hidden_dims_enc": [200, 200],
            "FC_hidden_dims_dec": [200, 200],
        },
        "estimator_name": estimator_name,
        "lambda_1": 2 / 3,
        "lambda_2": 1.0,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "SEED": 42,
        "num_epochs": 150,
        "log_every_k_epochs": 1,
    }
    if estimator_name == "NVIL":
        config["Baseline-Setup"] = {
            "input_dim": 1 * 28 * 28,
            "FC_hidden_dims": [400],
            "output_dim": 1,
        }
        config["tune_lr"] = 0.01
        config["tune_weight_decay"] = 1e-7
    elif estimator_name == "REBAR":
        config["tune_lr"] = 1e-6 
        config["tune_weight_decay"] = 1e-7
        config["eta"] = 1.0
        config["log_temp_init"] = -2.0
        config["VAE-Setup"]["fixed_var"] = 0.1
    elif estimator_name == "RELAX":
        config["tune_lr"] = 0.1
        config["tune_weight_decay"] = 1e-6
        config["C_PHI-Setup"] = {
            "input_dim": 3,
            "output_dim": 3,
            "FC_hidden_dims": [200, 200],
            "log_temp_init": 0.0,
        }
        # config["VAE-Setup"]["fixed_var"] = 1

    return config


if __name__ == "__main__":
    run_experiment()