from importlib.metadata import requires
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import torchvision as tv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from strimadec.models.modules import VAE, BaselineNet
from strimadec.models.utils import DVAE_LossModel, compute_accuracy
from strimadec.datasets import MNIST
from strimadec.discrete_gradient_estimators import analytical
from strimadec.discrete_gradient_estimators import REINFORCE, NVIL, CONCRETE, REBAR, RELAX
from strimadec.discrete_gradient_estimators.utils import set_requires_grad


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
        self.estimator_name, self.dataset_name = config["estimator_name"], config["dataset_name"]
        self.lr, self.weight_decay = config["lr"], config["weight_decay"]
        self.log_every_k_epochs = config["log_every_k_epochs"]
        # define model
        self.model = DVAE_LossModel(discrete_VAE)
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
        self.example_input_array = torch.Tensor(*self.model.VAE.example_input_shape)
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
        results = self.model.VAE(x)
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
        return losses_dict

    def training_step(self, batch, batch_idx):
        """the actual training step of the model happens here"""
        optimizer = self.optimizers()
        optimizer.zero_grad()
        x, labels = batch  # labels are only used for logging
        # loss computation
        losses = self.compute_loss(x)
        # model_params = list(self.model.VAE.parameters())
        KL_Div, NLL, NLL_estimator = losses["KL-Div"], losses["NLL"], losses["NLL_estimator"]
        # compute loss and exclude surrogate NLL_estimator loss in numerical representation
        loss = (KL_Div + NLL + NLL_estimator - NLL_estimator.detach()).mean()
        # actual training step, i.e., backpropagation
        if self.estimator_name == "RELAX":  # fix c_phi for estimator backward
            set_requires_grad(self.c_phi, False)

        self.manual_backward(loss, optimizer)

        if self.estimator_name == "RELAX":  # unfix to allow for updates of c_phi
            set_requires_grad(self.c_phi, True)
        optimizer.step()
        # log losses
        self.log("losses/loss", loss, on_step=False, on_epoch=True)
        self.log("losses/KL-Div", KL_Div.mean(), on_step=False, on_epoch=True)
        self.log("losses/NLL", NLL.mean(), on_step=False, on_epoch=True)
        self.log("losses/NLL_estimator", NLL_estimator.mean(), on_step=False, on_epoch=True)
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
        if (step + 1) % self.log_every_k_epochs == 0:
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

        return

    ########################################
    ######### VALIDATION FUNCTIONS #########
    ########################################

    # def validation_step(self, val_batch, batch_idx):
    #     x, labels = val_batch  # labels are not used here (unsupervised)
    #     # inference step
    #     results = self.forward(x)
    #     # loss unscaled (beta=1)
    #     return {"loss_unscaled": loss_unscaled}

    ########################################
    ####### PLOT AND HELPER FUNCTIONS ######
    ########################################

    def create_reconstructions(self, image_batch, n_samples):
        i_samples = np.random.choice(range(len(image_batch)), n_samples, False)
        images = image_batch[i_samples]

        results = self.forward(images)
        x_tilde = results["x_tilde"]
        if self.model.VAE.decoder_distribution == "Gaussian":
            x_tilde = x_tilde.clamp(0, 1)

        images_cat = torch.cat((images.unsqueeze(0), x_tilde.unsqueeze(0)), 0)
        return images_cat

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
        if self.dataset_name == "MNIST":

            # self.dataset = MNIST(train=True, download=True)
            MNIST_dataset = datasets.MNIST(
                "./data", transform=transforms.ToTensor(), train=True, download=True
            )
            # TEMPORARY
            from PIL import Image, ImageDraw
            from sklearn.preprocessing import OneHotEncoder
            from torch.utils.data import TensorDataset
            import numpy as np

            # select only specific digits
            data = []
            labels = []
            for digit in [2, 6, 9]:
                indices_digits = MNIST_dataset.targets == digit
                torch_imgs = [
                    transforms.ToTensor()(Image.fromarray(img.numpy(), mode="L"))
                    for img in MNIST_dataset.data[indices_digits]
                ]
                data.append(torch.vstack(torch_imgs))
                labels.extend([digit] * sum(indices_digits))
            # vertical stack torch tensors within data list
            data = torch.vstack(data).unsqueeze(1)
            # create one-hot encoded labels
            labels = OneHotEncoder().fit_transform(np.array(labels).reshape(-1, 1)).toarray()
            # make tensor dataset
            self.dataset = TensorDataset(data, torch.from_numpy(labels))
        return

    # def setup(self, stage=None):
    #     # Assign train/val datasets for use in dataloaders
    #     num_samples_val1 = int(0.8 * len(self.dataset))
    #     num_samples_val2 = len(self.dataset) - num_samples_val1
    #     split = [num_samples_val1, num_samples_val2]
    #     self.train_ds, self.val_ds = random_split(self.dataset, split)
    #     return

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1000, num_workers=12, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.dataset, batch_size=64, num_workers=12, shuffle=False)