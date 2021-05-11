import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from strimadec.models import DVAE
from strimadec.experiments.single_object_multi_class.utils import get_DVAE_config


def run_experiment(estimator_name, num_epochs, ds_name, decoder_dist, num_clusters, SEED):
    config = get_DVAE_config(ds_name, num_epochs, estimator_name, decoder_dist, num_clusters, SEED)
    # make experiment reproducible
    seed_everything(config["SEED"])
    # instantiate model
    model = DVAE(config)
    # define logger
    logger = TensorBoardLogger("DVAE", name=config["experiment_name"])
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        "--estimator_name",
        default="REINFORCE",
        choices=["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "Exact gradient"],
        action="store",
        type=str,
        help="name of gradient estimator",
    )
    parser.add_argument(
        "--decoder_dist",
        default="Gaussian",
        choices=["Gaussian", "Bernoulli"],
        action="store",
        type=str,
        help="asssumed decoder distribution",
    )
    parser.add_argument(
        "--dataset_name",
        default="SimplifiedMNIST",
        choices=["SimplifiedMNIST", "FullMNIST"],
        action="store",
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--num_clusters", default=3, action="store", type=int, help="latent dimension of VAE"
    )

    parse_results = parser.parse_args()

    num_epochs = parse_results.num_epochs
    num_repetitions = parse_results.num_repetitions
    estimator_name = parse_results.estimator_name
    decoder_dist = parse_results.decoder_dist
    ds_name = parse_results.dataset_name
    num_clusters = parse_results.num_clusters

    print(f"{estimator_name}-Experiment with {ds_name} Dataset")
    for i in range(num_repetitions):
        print(f"Start {i+1}/{num_repetitions}")
        run_experiment(
            ds_name=ds_name,
            num_epochs=num_epochs,
            estimator_name=estimator_name,
            decoder_dist=decoder_dist,
            num_clusters=num_clusters,
            SEED=i,
        )
