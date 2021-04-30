import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms

from strimadec.models.modules import VAE
from strimadec.datasets import MNIST
from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR, RELAX


class DVAE(pl.LightningModule):

    """
        D-VAE lighting module, i.e., a VAE with discrete latent space (categorical distribution)
        in which a discrete gradient estimator is used to approximate the the discrete expectation

    Args:
        config (dict): dictionary containing the following entries
            VAE-Setup (dict): dictionary containing the setup (hidden layers, etc) of the VAE
            dataset_name (str): name of dataset
            estimator_name (str): name of the discrete gradient estimator
            lr (float): learning rate for VAE network parameters (ADAM)
            weight_decay (float): weight_decay for VAE network parameters (ADAM)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        assert (
            config["VAE-Setup"]["encoding_distribution"] == "Categorical"
        ), "The encoding distribution needs to be set to `Categorical` for the DVAE"
        # parse config
        self.VAE = VAE(config["VAE-Setup"])
        self.estimator_name, self.dataset_name = config["estimator_name"], config["dataset_name"]
        self.lr, self.weight_decay = config["lr"], config["weight_decay"]
        # define prior_probs (uniform)
        self.prior_probs = (1 / self.VAE.latent_dim) * torch.ones(self.VAE.latent_dim)
        if not any(e in self.estimator_name for e in ["REINFORCE", "Exact gradient", "CONCRETE"]):
            self.tune_lr = config["tune_lr"]
            self.tune_weight_decay = config["tune_weight_decay"]

        return

    def forward(self, x):
        """
            defines inference procedure of the D-VAE, i.e., computes latent space and
            keeps track of useful metrics

        Args:
            x (torch tensor): image [batch, img_channels, img_dim, img_dim]

        Returns:
            results (dict): results dictionary containing
                x_tilde (torch tensor): image reconstruction [batch, img_channels, img_dim, img_dim]
                z (torch tensor): latent space - one hot encoded [batch, L]
                probs_logits (torch tensor): assigned probability of latents in logits [batch, L]
        """
        results = self.VAE(x)
        return results

    ########################################
    ########## TRAINING FUNCTIONS ##########
    ########################################

    def compute_loss(self, x):
        # get assigned probability of latents in logits [batch, L]
        probs_logits = self.VAE.encode(x)
        # compute KL divergence loss [batch]
        latent_dist = torch.distributions.OneHotCategorical(logits=probs_logits)
        latent_prior_dists = torch.distributions.OneHotCategorical(probs=self.prior_probs)
        KL_Div = torch.distributions.kl_divergence(latent_dist, latent_prior_dists)
        # compute negative log-likelihood [batch] (no gradients for encoder network)

        # compute NLL using discrete gradient estimator [batch] (gradients for encoder network)
        if self.estimator_name == "Exact gradient":
            NLL_estimator = analytical(probs_logits, x, loss_func)
        elif self.estimator_name == "REINFORCE":
            NLL_estimator = REINFORCE(probs_logits, x, loss_func)
        elif self.estimator_name == "NVIL":
            baseline_vals = self.baseline_net.forward(x)
            NLL_estimator = NVIL(probs_logits, x, baseline_vals, loss_func)
        elif self.estimator_name == "CONCRETE":
            NLL_estimator = CONCRETE(probs_logits, x, self.temp, loss_func)
        elif self.estimator_name == "REBAR":
            NLL_estimator = REBAR(probs_logits, x, self.log_temp.exp(), self.eta, loss_func)
        elif self.estimator_name == "RELAX":
            NLL_estimator = RELAX(probs_logits, x, self.c_phi, loss_func)
        results = {"KL_Div": KL_Div, "NLL": NLL, "NLL_estimator": NLL_estimator}
        return results

    ########################################
    ######### TRAINING SETUP HOOKS #########
    ########################################

    @property
    def automatic_optimization(self) -> bool:
        return False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.VAE.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    ########################################
    ########## DATA RELATED HOOKS ##########
    ########################################

    def prepare_data(self) -> None:
        if self.dataset_name == "MNIST":
            self.dataset = MNIST(train=True, download=True)
        return

    # def setup(self, stage=None):
    #     # Assign train/val datasets for use in dataloaders
    #     num_samples_val1 = int(0.8 * len(self.dataset))
    #     num_samples_val2 = len(self.dataset) - num_samples_val1
    #     split = [num_samples_val1, num_samples_val2]
    #     self.train_ds, self.val_ds = random_split(self.dataset, split)
    #     return

    # def train_dataloader(self):
    #     return DataLoader(self.train_ds, batch_size=64, num_workers=12, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_ds, batch_size=64, num_workers=12, shuffle=False)