import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from strimadec.models import AIR, DAIR
from strimadec.experiments.multi_object_multi_class.utils import build_AIR_config, build_DAIR_config
from strimadec.experiments.multi_object_multi_class.utils.experimental_setup import (
    build_AIR_config, build_DAIR_config
)

def run_single_experiment(model_name, num_epochs, dataset_name, num_clusters, store_dir, SEED):
    # make experiment reproducible
    seed_everything(SEED)
    # instantiate model
    if model_name == "AIR":
        config = build_AIR_config(dataset_name, num_epochs, num_clusters, SEED)
        model = AIR(config)
    elif model_name == "DAIR":
        config = build_DAIR_config(dataset_name, num_epochs, num_clusters, SEED)
        model = DAIR(config)
    # define logger
    store_dir = os.path.join(store_dir, f"Comparison/{model_name}_results")
    logger = TensorBoardLogger(store_dir, name=config["experiment_name"])
    # define callback of model checkpoint
    checkpoint_callback = ModelCheckpoint(period=1)
    # initialize pytorch lightning trainer
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        deterministic=True,
        gpus=num_gpus,
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