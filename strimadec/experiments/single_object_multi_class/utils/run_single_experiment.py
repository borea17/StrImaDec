import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from strimadec.models import DVAE, DVAEST
from strimadec.experiments.single_object_multi_class.utils import (
    experimental_setup,
    localization_setup,
)


def run_single_experiment(
    model_name,
    estimator_name,
    num_epochs,
    dataset_name,
    decoder_dist,
    num_clusters,
    store_dir,
    SEED,
):
    """
        executes the training for a specific model

    Args:
        model_name (str): name of the model ("DVAE" or "DVAEST")
        estimator_name (str): name of the gradient estimator to be used
        num_epochs (int): number of epochs to train
        dataset_name (str): name of the dataset ("SimplifiedMNIST", "FullMNIST", "Letters")
        decoder_dist (str): decoder distribution ("Gaussian", "Bernoulli")
        num_clusters (int): latent dimensionality
        store_dir (str): path to parent directory in which results will be stored
        SEED (int): seed for making experiment reproducible
    """
    # make experiment reproducible
    seed_everything(SEED)
    config = experimental_setup(
        dataset_name, num_epochs, estimator_name, decoder_dist, num_clusters, SEED
    )
    # instantiate model
    if model_name == "D-VAE":
        model = DVAE(config)
    elif model_name == "D-VAE-ST":
        config["Localization-Setup"] = localization_setup()
        model = DVAEST(config)
    # define logger
    store_dir = os.path.join(store_dir, f"{model_name}_results")
    logger = TensorBoardLogger(store_dir, name=config["experiment_name"])
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