import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
from torch.utils.data import DataLoader

from strimadec.datasets import SimplifiedMNIST, FullMNIST, Letters, FashionMNIST
from strimadec.models.modules import VAE, BaselineNet, LocalizationNet
from strimadec.models.utils import DVAEST_LossModel, compute_accuracy
from strimadec.datasets import FullMNIST
from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR, RELAX
from strimadec.discrete_gradient_estimators.utils import set_requires_grad


class DVAEST(pl.LightningModule):

    """
        D-VAE-ST lighting module, i.e., a VAE with discrete latent space (categorical distribution)
        in which a discrete gradient estimator is used to approximate the the discrete expectation
        and a Spatial Transformer (ST) is used to include positional & shape variations of the data

    Args:
        config (dict): dictionary containing the following entries
            VAE-Setup (dict): dictionary containing the setup (hidden layers, etc) of the VAE
            Localization-Setup (dict): dictionary for neural network that estimates ST parameters
            dataset_name (str): name of dataset
            estimator_name (str): name of the discrete gradient estimator
            lr (float): learning rate for VAE network parameters (ADAM)
            weight_decay (float): weight_decay for VAE network parameters (ADAM)
            ################## DEPENDENT PARAMS ##################
            estimator_name == "NVIL":
                tune_lr (float): learning rate for baseline network parameters (ADAM)
                tune_weight_decay (float): weight_decay for baseline network parameters (ADAM)
                Baseline-Setup (dict): dictionary containing the setup of the neural baseline
            estimator_name == "CONCRETE":
                lambda_1 (float): temperature of concrete distribution [latent dist]
                lambda_2 (float): temperature of concrete distribution [latent prior dist]
            estimator_name == "REBAR":
                tune_lr (float): learning rate for baseline network parameters (ADAM)
                tune_weight_decay (float): weight_decay for baseline network parameters (ADAM)
                eta (float): hyperparameter that scales control variate [1]
                log_temp_init (float): tuneable hyperparameter (temperature of concrete) [1]
            estimator_name == "RELAX":
                tune_lr (float): learning rate for baseline network parameters (ADAM)
                tune_weight_decay (float): weight_decay for baseline network parameters (ADAM)
                C_PHI-Setup (dict): dictionary containing the setup of C_PHI network
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        assert (
            config["VAE-Setup"]["encoder_distribution"] == "Categorical"
        ), "The encoding distribution needs to be set to `Categorical` for the DVAE"
        # parse config
        discrete_VAE = VAE(config["VAE-Setup"])
        localization_net = LocalizationNet(config["Localization-Setup"])
        self.estimator_name, self.dataset_name = config["estimator_name"], config["dataset_name"]
        self.lr, self.weight_decay = config["lr"], config["weight_decay"]
        self.log_every_k_epochs = config["log_every_k_epochs"]
        # define model
        self.model = DVAEST_LossModel(discrete_VAE, localization_net)
        # define prior_probs (uniform)
        prior_probs = (1 / self.model.VAE.latent_dim) * torch.ones(self.model.VAE.latent_dim)
        self.register_buffer("prior_probs", prior_probs.clone())
        if self.estimator_name in ["NVIL", "REBAR", "RELAX"]:
            self.tune_lr = config["tune_lr"]
            self.tune_weight_decay = config["tune_weight_decay"]
        else:
            self.tuneable_hyperparameters = []
        # estimator dependent parsing
        if self.estimator_name == "CONCRETE":
            self.register_buffer("lambda_1", torch.tensor(config["lambda_1"]))
            self.register_buffer("lambda_2", torch.tensor(config["lambda_1"]))
        elif self.estimator_name == "NVIL":
            self.baseline_net = BaselineNet(config["Baseline-Setup"])
            self.tuneable_hyperparameters = self.baseline_net.parameters()
        elif self.estimator_name == "REBAR":
            self.eta = config["eta"]
            self.log_temp = nn.Parameter(torch.tensor(config["log_temp_init"]), requires_grad=True)
            self.tuneable_hyperparameters = [self.log_temp]
        elif self.estimator_name == "RELAX":
            self.c_phi = BaselineNet(config["C_PHI-Setup"])
            self.tuneable_hyperparameters = self.c_phi.parameters()
        # input and output sizes infered by pytorch lightning
        self.example_input_array = torch.zeros(*self.model.VAE.example_input_shape)
        return

    def forward(self, x):
        """
            defines inference procedure of the D-VAE-ST, i.e., computes latent space and
            keeps track of useful metrics

        Args:
            x (torch tensor): image [batch, img_channels, img_dim, img_dim]

        Returns:
            results (dict): results dictionary containing
                x_p (torch tensor): image prototype [batch, img_channels, img_dim, img_dim]
                x_tilde (torch tensor): image reconstruction [batch, img_channels, img_dim, img_dim]
                z (torch tensor): classlatent space - one hot encoded [batch, L]
                probs_logits (torch tensor): assigned probability of latents in logits [batch, L]
                t (torch tensor): sampled transformation vector [batch, 6]
                mu_t (torch tensor): mean of sampled transformation t [batch, 6]
                log_var_t (torch tensor): log variance of sampled transformation t [batch, 6]
        """
        results_VAE = self.model.VAE(x)
        results_localization = self.model.localization_net(x)
        x_p = results_VAE["x_tilde"]
        x_tilde = self.model.localization_net.spatial_transform(x_p, results_localization["t"])
        results = {
            "x_p": x_p,
            "x_tilde": x_tilde,
            "z": results_VAE["z"],
            "probs_logits": results_VAE["probs_logits"],
            "t": results_localization["t"],
            "mu_t": results_localization["mu_t"],
            "log_var_t": results_localization["log_var_t"],
        }
        return results

    ########################################
    ########## TRAINING FUNCTIONS ##########
    ########################################

    def compute_loss(self, x):
        # get assigned probability of latents in logits [batch, L]
        probs_logits = self.model.VAE.encode(x)
        # compute KL divergence loss [batch]
        if self.estimator_name == "CONCRETE":  # estimate through sampling
            prior_probs, lambda_1, lambda_2 = self.prior_probs, self.lambda_1, self.lambda_2
            latent_dist = dists.RelaxedOneHotCategorical(lambda_1, logits=probs_logits)
            z = latent_dist.rsample()
            log_prior_z = dists.RelaxedOneHotCategorical(lambda_2, probs=prior_probs).log_prob(z)
            log_q_z_g_x = latent_dist.log_prob(z)
            KL_Div = log_q_z_g_x - log_prior_z
        else:  # analytical solution
            latent_dist = dists.OneHotCategorical(logits=probs_logits)
            latent_prior_dist = dists.OneHotCategorical(probs=self.prior_probs)
            KL_Div = dists.kl_divergence(latent_dist, latent_prior_dist)
        # compute negative log-likelihood (no gradients for encoder net) and NLL_estimator [batch]
        loss_func = self.model.loss_func
        if self.estimator_name == "Exact gradient":  # no NLL_estimator needed here
            NLL = analytical(probs_logits, x, loss_func)
            NLL_estimator = torch.tensor(0.0).type(NLL.dtype)
        elif self.estimator_name == "REINFORCE":
            NLL_estimator, NLL = REINFORCE(probs_logits, x, loss_func)
        elif self.estimator_name == "NVIL":
            baseline_vals = self.baseline_net.forward(x)
            NLL_estimator, NLL = NVIL(probs_logits, x, baseline_vals, loss_func)
        elif self.estimator_name == "CONCRETE":  # no NLL_estimator needed here
            NLL = CONCRETE(probs_logits, x, self.lambda_1, loss_func)
            NLL_estimator = torch.tensor(0.0).type(NLL.dtype)
        elif self.estimator_name == "REBAR":
            eta, temp = self.eta, self.log_temp.exp()
            NLL_estimator, NLL = REBAR(probs_logits, x, temp, eta, self.model, loss_func)
        elif self.estimator_name == "RELAX":
            NLL_estimator, NLL = RELAX(probs_logits, x, self.c_phi, self.model, loss_func)
        losses_dict = {"KL-Div": KL_Div, "NLL": NLL, "NLL_estimator": NLL_estimator}
        if self.estimator_name == "NVIL":
            baseline_loss = (NLL - baseline_vals).pow(2).mean()
            losses_dict["baseline_loss"] = baseline_loss
        return losses_dict

    def training_step(self, batch, batch_idx):
        """the actual training step of the model happens here"""
        tic = time.time()

        optimizer = self.optimizers()
        optimizer.zero_grad()

        x, labels = batch  # labels are only used for logging
        # loss computation
        losses = self.compute_loss(x)
        KL_Div, NLL, NLL_estimator = losses["KL-Div"], losses["NLL"], losses["NLL_estimator"]
        KL_Div_pos = self.model.retrieve_kl_div_position()
        # compute loss and exclude surrogate NLL_estimator loss in numerical representation
        loss = (KL_Div + NLL + NLL_estimator - NLL_estimator.detach() + KL_Div_pos).mean()
        # actual training step, i.e., backpropagation
        if self.estimator_name == "REBAR":  # fix log_temp for estimator backward
            self.log_temp.requires_grad = False
        elif self.estimator_name == "RELAX":  # fix c_phi for estimator backward
            set_requires_grad(self.c_phi, False)

        loss.backward()

        if self.estimator_name == "REBAR":  # unfix to allow for updates of log_temp
            self.log_temp.requires_grad = True
        elif self.estimator_name == "RELAX":  # unfix to allow for updates of c_phi
            set_requires_grad(self.c_phi, True)

        optimizer.step()

        toc = time.time()
        # log losses
        if self.estimator_name == "NVIL":
            self.log("losses/baseline_loss", losses["baseline_loss"], on_step=False, on_epoch=True)
        self.log("losses/loss", loss, on_step=False, on_epoch=True)
        self.log("losses/KL-Div", KL_Div.mean(), on_step=False, on_epoch=True)
        self.log("losses/KL-Div-Position", KL_Div_pos.mean(), on_step=False, on_epoch=True)
        self.log("losses/NLL", NLL.mean(), on_step=False, on_epoch=True)
        self.log("losses/NLL_estimator", NLL_estimator.mean(), on_step=False, on_epoch=True)
        self.log("elapsed_times/time", toc-tic, on_step=False, on_epoch=True)
        # special logs
        if self.estimator_name == "REBAR":
            self.log("params/temp", self.log_temp.exp(), on_step=False, on_epoch=True)
        elif self.estimator_name == "RELAX":
            self.log("params/c_phi_temp", self.c_phi.log_temp.exp(), on_step=False, on_epoch=True)
        # define training step output for faster computation of accuracy
        training_step_output = {"x": x, "y": labels, "loss": loss}
        return training_step_output

    def training_epoch_end(self, training_step_outputs):
        """this function at the end of each training epoch"""
        step = self.current_epoch
        if (step + 1) % self.log_every_k_epochs == 0 or (step == 0):
            mode = "deterministic"
            acc, empty = compute_accuracy(training_step_outputs, self.model.VAE, mode)
            self.log(f"metrics/{mode}_accuracy", acc, on_step=False, on_epoch=True, logger=True)

            n_samples = 7
            last_image_batch = training_step_outputs[-1]["x"]
            # image and reconstructions
            img_recons = self.create_reconstructions(last_image_batch, n_samples)
            img_recons_cat = img_recons.view(-1, *last_image_batch.shape[1::])
            grid = tv.utils.make_grid(img_recons_cat, padding=10, pad_value=1, nrow=n_samples)
            self.logger.experiment.add_image("Image and Reconstruction", grid, step)
            # latent traversal
            img_latent_trav = self.create_latent_traversal()
            grid = tv.utils.make_grid(img_latent_trav, padding=10, pad_value=1, nrow=n_samples)
            self.logger.experiment.add_image("Latent Traversal", grid, step)
        return
    
    ########################################
    ######### EVALUATION FUNCTIONS #########
    ########################################

    def compute_test_loss(self, x):
        # get assigned probability of latents in logits [batch, L]
        probs_logits = self.model.VAE.encode(x)
        latent_dist = dists.OneHotCategorical(logits=probs_logits)
        latent_prior_dist = dists.OneHotCategorical(probs=self.prior_probs)
        KL_Div = dists.kl_divergence(latent_dist, latent_prior_dist)
        # sample class indices [batch, L]
        sampled_class = latent_dist.sample()
        # compute loss [batch]
        NLL = self.model.loss_func(sampled_class, x)
        KL_Div_position = self.model.retrieve_kl_div_position()
        losses_dict = {"KL-Div": KL_Div + KL_Div_position, "NLL": NLL}
        return losses_dict

    def compute_evaluation_metrics(self):
        # prepare data and retrieve dataloader
        self.prepare_data()
        train_dataloader = self.train_dataloader()
        N = len(train_dataloader)
        # initiliaze
        training_step_outputs = []
        NLLs, KL_Divs = np.zeros(N), np.zeros(N)
        for i, (x, y) in enumerate(train_dataloader):
            losses_dict = self.compute_test_loss(x)
            NLLs[i] = losses_dict["NLL"].mean().item()
            KL_Divs[i] = losses_dict["KL-Div"].mean().item()
            training_step_outputs.append({"x": x, "y": y})
        acc, empty = compute_accuracy(training_step_outputs, self.model.VAE, "deterministic")
        results = {
            "NLL": NLLs.mean(),
            "KL_Div": KL_Divs.mean(),
            "acc": acc,
            "empty": empty
        }
        return results

    ########################################
    ####### PLOT AND HELPER FUNCTIONS ######
    ########################################

    def create_reconstructions(self, image_batch, n_samples):
        i_samples = np.random.choice(range(len(image_batch)), n_samples, False)
        images = image_batch[i_samples]

        results = self.forward(images)
        x_p = results["x_p"].clamp(0, 1)
        x_tilde = results["x_tilde"].clamp(0, 1)

        images_cat = torch.cat((images.unsqueeze(0), x_p.unsqueeze(0), x_tilde.unsqueeze(0)), 0)
        return images_cat

    def create_latent_traversal(self):
        """in the discrete case there are only latent_dim possible latent states"""
        latent_dim = self.model.VAE.latent_dim
        img_channels, img_dim = self.model.VAE.img_channels, self.model.VAE.img_dim
        images = torch.zeros([latent_dim, img_channels, img_dim, img_dim], device=self.device)
        identity = torch.eye(latent_dim, device=self.device)
        for i_lat in range(latent_dim):
            z = identity[i_lat].unsqueeze(0)  # [1, latent_dim]
            images[i_lat] = self.model.VAE.decode(z).squeeze(0).clamp(0, 1)
        return images

    def plot_reconstructions(self, store_path_fig):
        """figures for thesis"""
        n_samples = 7
        i_samples = np.random.choice(range(len(self.dataset)), n_samples, replace=False)
        fig = plt.figure(figsize=(12, 5))
        for counter, i_sample in enumerate(i_samples):
            orig_img = self.dataset[i_sample][0]
            # plot original img
            ax = plt.subplot(3, n_samples, 1 + counter)
            plt.imshow(tv.transforms.ToPILImage()(orig_img), vmin=0, vmax=256, cmap="gray")
            plt.axis("off")
            if counter == 0:
                ax.annotate("Input Image", xy=(-0.1, 0.5), xycoords="axes fraction", 
                             va="center", ha="right", fontsize=14)
            results = self.forward(orig_img.unsqueeze(0).to(self.device))
            x_p = results["x_p"].clamp(0, 1).squeeze(0)
            x_tilde = results["x_tilde"].clamp(0, 1).squeeze(0)
            # plot prototype
            ax = plt.subplot(3, n_samples, 1 + n_samples + counter)
            plt.imshow(tv.transforms.ToPILImage()(x_p), vmin=0, vmax=256, cmap="gray")
            plt.axis("off")
            if counter == 0:
                ax.annotate("Prototype", xy=(-0.1, 0.5), xycoords="axes fraction", 
                             va="center", ha="right", fontsize=14)
            # plot reconstruction
            ax = plt.subplot(3, n_samples, 1 + 2*n_samples + counter)
            plt.imshow(tv.transforms.ToPILImage()(x_tilde), vmin=0, vmax=256, cmap="gray")
            plt.axis("off")
            if counter == 0:
                ax.annotate("Reconstruction", xy=(-0.1, 0.5), xycoords="axes fraction", 
                             va="center", ha="right", fontsize=14)
        plt.show()
        # store figurej
        fig.savefig(store_path_fig, bbox_inches="tight")
        return

    def plot_latents(self, store_path_fig):
        """figures for thesis"""
        latent_dim = self.model.VAE.latent_dim
        if latent_dim < 5:
            num_rows, num_cols = 1, latent_dim
        elif latent_dim < 15:
            num_rows, num_cols = 2, np.ceil(latent_dim/2)
        else:
            num_rows, num_cols = 4, 7
        fig = plt.figure(figsize=(12, 5))
        images = self.create_latent_traversal()
        for row_index in range(latent_dim):
            ax = plt.subplot(num_rows, num_cols, row_index + 1)
            
            plt.imshow(tv.transforms.ToPILImage()(images[row_index]), vmin=0, vmax=256, cmap="gray")
            plt.axis('off')
        plt.show()
        # store figure
        fig.savefig(store_path_fig, bbox_inches="tight")
        return

    ########################################
    ######### TRAINING SETUP HOOKS #########
    ########################################

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        if self.tuneable_hyperparameters:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.model.parameters(),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": self.tuneable_hyperparameters,
                        "lr": self.tune_lr,
                        "weight_decay": self.tune_weight_decay,
                    },
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return optimizer

    ########################################
    ########## DATA RELATED HOOKS ##########
    ########################################

    def prepare_data(self) -> None:
        if self.dataset_name == "SimplifiedMNIST":
            self.dataset = SimplifiedMNIST(train=True, digits=[2, 6, 9])
        elif self.dataset_name == "FullMNIST":
            self.dataset = FullMNIST(train=True)
        elif self.dataset_name == "Letters":
            self.dataset = Letters()
        elif self.dataset_name == "FashionMNIST":
            self.dataset = FashionMNIST(train=True)
        return

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1000, num_workers=12, shuffle=True)
