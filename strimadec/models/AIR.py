import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import torch.distributions as dists
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from strimadec.datasets.multi_object_dataset import MultiMNIST
from strimadec.models.modules import AIR_BaseClass, VAE, RNN
from strimadec.models.utils import gaussian_kl, bernoulli_kl


class AIR(AIR_BaseClass):

    """

        Attend-Infer-Repeat class as described by Eslami et al. (2016)

    Args:
        config (dict): dictionary containing the following configurations
            What VAE-Setup (dict): dictionary containing the setup for VAE of AIR
            RNN-Setup (dict): dictionary containing the setup of the RNN for z_pres, z_where
            RNN Baseline-Setup (dict): dictionary containing the setup for the RNN NVIL baseline
            dataset_name (str): name of dataset
            number_of_slots_train (int): maximum number of objects assumed during training
            img_shape (list): shape of whole image provided as a list
            prior_z_pres (float): prior on z_pres
            prior_mean_z_where (list): prior on mean of z_where
            prior_var_z_where (list): prior on var of z_where
            lr (float): learning rate for VAE network parameters (ADAM)
            weight_decay (float): weight_decay for VAE network parameters (ADAM)
            base_lr (float): learning rate for baseline network parameters (ADAM)
            base_weight_decay (float): weight_decay for baseline network parameters (ADAM)
    """

    def __init__(self, config) -> None:
        super().__init__()
        # store config in logger
        self.save_hyperparameters(config)
        # parse config
        self.vae = VAE(config["What VAE-Setup"])
        self.rnn = RNN(config["RNN-Setup"])
        self.baseline = RNN(config["RNN Baseline-Setup"])

        self.dataset_name = config["dataset_name"]
        self.N_train = config["number_of_slots_train"]
        self.lr, self.weight_decay = config["lr"], config["weight_decay"]
        self.base_lr, self.base_weight_decay = config["base_lr"], config["base_weight_decay"]
        self.log_every_k_epochs = config["log_every_k_epochs"]
        self.img_shape = config["img_shape"]
        self.prior_z_pres = config["prior_z_pres"]
        self.register_buffer("prior_mean_z_where", torch.tensor(config["prior_mean_z_where"]))
        self.register_buffer("prior_var_z_where", torch.tensor(config["prior_var_z_where"]))
        # store some useful parameters as attributes
        self.window_dim = self.vae.img_dim
        self.z_what_dim = self.vae.latent_dim
        self.omega_dim = 1 + 2 * 3 + 2 * self.z_what_dim
        self.z_dim = 1 + 3 + self.z_what_dim
        self.rnn_hidden_state_dim = config["RNN-Setup"]["hidden_state_dim"]
        self.rnn_hidden_state_dim_b = config["RNN Baseline-Setup"]["hidden_state_dim"]
        return

    def forward(self, x, N, save_attention_rectangle=False):
        """
        defines the inference procedure of AIR, i.e., computes the latent space
        and keeps track of useful metrics

        Args:
            x (torch.tensor): image [batch_size, img_channels, img_dim, img_dim]
            N (int): maximum number of inference steps

        Returns:
            results (dict) output dictionary containing
                all_z_pres (torch tensor): sampled z_pres for each N [batch, N]
                z_pres_likelihood (torch tensor): likelihood of sampled z_pres for each N [batch, N]
                mask_delay (torch tensor): z_pres delayed by one [batch, N]
                all_prob_pres (torch tensor): mean of z_pres for each N [batch, N, 1]
                all_z_where (torch tensor): sampled z_where for each N [batch, N, 3]
                all_mu_where (torch tensor): mean of z_where for each N [batch, N, 3]
                all_log_var_where (torch tensor): log variance z_where for each N [batch, N, 3]
                all_z_what (torch tensor): sampled z_what for each N [batch, N, z_what_dim]
                all_mu_what (torch tensor): mean z_what for each N [batch, N, z_what_dim]
                all_log_var_what (torch tensor): log variance z_what for each N [batch, N, z_what_dim]
                baseline_values (torch tensor): neural baseline value for NLL [batch, N]
                x_tilde (torch tensor): reconstruction of x based on latent space [x.shape]
                counts (torch tensor): number of identified entities per image [batch, N]
        """
        batch_size = x.shape[0]
        # initializations
        all_z = torch.empty((batch_size, N, self.z_dim), device=x.device)
        z_pres_likelihood = torch.empty((batch_size, N), device=x.device)
        mask_delay = torch.empty((batch_size, N), device=x.device)
        all_omega = torch.empty((batch_size, N, self.omega_dim), device=x.device)
        all_x_tilde = torch.empty((batch_size, N, *self.img_shape), device=x.device)
        baseline_values = torch.empty((batch_size, N), device=x.device)

        z_im1 = torch.ones((batch_size, self.z_dim), device=x.device)
        h_im1 = torch.zeros((batch_size, self.rnn_hidden_state_dim), device=x.device)
        h_im1_b = torch.zeros((batch_size, self.rnn_hidden_state_dim_b), device=x.device)
        if save_attention_rectangle:
            attention_rects = torch.empty((batch_size, N, 2, 5), device=x.device)
        for i in range(N):
            z_im1_pres = z_im1[:, 0:1]
            # mask delay is used to zero out all steps AFTER FIRST z_pres = 0
            mask_delay[:, i] = z_im1_pres.squeeze(1)
            # obtain parameters of sampling distribution and hidden state
            omega_i, h_i = self.rnn(x, z_im1, h_im1)
            # baseline version
            baseline_i, h_i_b = self.baseline(x.detach(), z_im1.detach(), h_im1_b)
            # set baseline 0 if z_im1_pres = 0
            baseline_value = (baseline_i * z_im1_pres).squeeze()
            # extract sample distributions parameters from omega_i
            prob_pres_i = omega_i[:, 0:1]
            mu_where_i = omega_i[:, 1:4]
            log_var_where_i = omega_i[:, 4:7]
            # sample from distribution to obtain z_i_pres and z_i_where
            z_i_pres_post = dists.Bernoulli(probs=prob_pres_i)
            z_i_pres = z_i_pres_post.sample() * z_im1_pres
            # likelihood of sampled z_i_pres (only if z_im_pres = 1)
            z_pres_likelihood[:, i] = (z_i_pres_post.log_prob(z_i_pres) * z_im1_pres).squeeze(1)
            # get z_i_where by reparametrization trick
            epsilon_w = torch.randn_like(log_var_where_i)
            z_i_where = mu_where_i + torch.exp(0.5 * log_var_where_i) * epsilon_w
            # use z_where and x to obtain x_att_i
            x_att_i = AIR_BaseClass.image_to_window(
                x, z_i_where, self.img_shape[0], self.window_dim
            )
            # put x_att_i through VAE
            results_vae = self.vae(x_att_i)
            x_tilde_att_i = results_vae["x_tilde"]
            z_i_what = results_vae["z"]
            mu_what_i, log_var_what_i = results_vae["mu_E"], results_vae["log_var_E"]
            # create image reconstruction
            x_tilde_i = AIR_BaseClass.window_to_image(x_tilde_att_i, z_i_where, self.img_shape)
            # update im1 with current versions
            z_im1 = torch.cat((z_i_pres, z_i_where, z_i_what), 1)
            h_im1 = h_i
            h_im1_b = h_i_b
            # put all distribution parameters into omega_i
            omega_i = torch.cat(
                (prob_pres_i, mu_where_i, log_var_where_i, mu_what_i, log_var_what_i), 1
            )
            # store intermediate results
            all_z[:, i : i + 1] = z_im1.unsqueeze(1)
            all_omega[:, i : i + 1] = omega_i.unsqueeze(1)
            all_x_tilde[:, i] = x_tilde_i
            baseline_values[:, i] = baseline_value
            # for nice visualization
            if save_attention_rectangle:
                attention_rects[:, i] = AIR_BaseClass.get_attention_rectangle(
                    z_i_where, self.img_shape[1]
                ) * z_i_pres.unsqueeze(1)
        # save results in dict (easy accessibility)
        results = dict()
        results["z_pres_likelihood"] = z_pres_likelihood
        results["all_z_pres"] = all_z[:, :, 0:1]
        results["mask_delay"] = mask_delay
        results["all_prob_pres"] = all_omega[:, :, 0:1]
        results["all_z_where"] = all_z[:, :, 1:4]
        results["all_mu_where"] = all_omega[:, :, 1:4]
        results["all_log_var_where"] = all_omega[:, :, 4:7]
        results["all_z_what"] = all_z[:, :, 4::]
        results["all_mu_what"] = all_omega[:, :, 7 : 7 + self.z_what_dim]
        results["all_log_var_what"] = all_omega[:, :, 7 + self.z_what_dim : :]
        results["baseline_values"] = baseline_values
        if save_attention_rectangle:
            results["attention_rects"] = attention_rects
        results["x_tilde_i"] = all_x_tilde
        # compute reconstructed image (take only x_tilde_i with z_i_pres=1)
        results["x_tilde"] = (all_z[:, :, 0:1].unsqueeze(3).unsqueeze(3) * all_x_tilde).sum(axis=1)
        # compute counts as identified objects (sum z_i_pres)
        results["counts"] = results["all_z_pres"].sum(1).to(dtype=torch.long)
        return results

    ########################################
    #########  TRAINING FUNCTIONS  #########
    ########################################

    def training_step(self, batch, batch_idx):
        """compute loss of AIR (essentially a VAE loss)
        assuming the following prior distributions for the latent variables

            z_where ~ N (prior_mean_where, prior_var_where)
            z_what ~ N (0, 1)
            z_pres ~ Bern (prior_p_pres)
        """
        x, labels = batch  # labels are only used for logging
        # inference step
        results = self.forward(x, self.N_train)
        ############################# LOSS COMPUTATION #############################
        # use mask delay to zero out irrelevants
        mask_delay = results["mask_delay"]
        # kl div for z_pres (between two Bernoulli distributions) [batch, N]
        q_z_pres = results["all_prob_pres"]  # [batch, N, 1]
        p_z_pres = self.prior_z_pres * torch.ones_like(q_z_pres)  # [batch, N, 1]
        kl_div_pres = bernoulli_kl(q_z_pres, p_z_pres).sum(axis=2) * mask_delay
        # kl div for z_what (standard VAE regularization term) [batch, N]
        q_z_what = [results["all_mu_what"], results["all_log_var_what"].exp()]
        p_z_what = [torch.zeros_like(q_z_what[0]), torch.ones_like(q_z_what[0])]
        kl_div_what = gaussian_kl(q_z_what, p_z_what).sum(axis=2) * mask_delay
        # kl div for z_where (between two Gaussians)  [batch, N]
        q_z_where = [results["all_mu_where"], results["all_log_var_where"].exp()]
        p_mu_where = self.prior_mean_z_where.expand(q_z_where[0].shape)
        p_var_where = self.prior_var_z_where.expand(q_z_where[0].shape)
        p_z_where = [p_mu_where, p_var_where]
        kl_div_where = gaussian_kl(q_z_where, p_z_where).sum(axis=2) * mask_delay
        # compute batch-wise kl div [batch]
        kl_div = (kl_div_pres + kl_div_what + kl_div_where).sum(1)
        # NLL for Gaussian decoder (no gradient for z_pres)
        factor = 0.5 * (1 / self.vae.fixed_var)
        NLL = factor * ((x - results["x_tilde"]) ** 2).sum(axis=(1, 2, 3))
        # NVIL estimator for NLL (gradient of z_pres)
        baseline_target = NLL.unsqueeze(1)
        NVIL_term = (
            (baseline_target - results["baseline_values"]).detach()
            * results["z_pres_likelihood"]
            * mask_delay
        ).sum(1)
        # baseline model loss
        baseline_loss = (
            (results["baseline_values"] - baseline_target.detach()) ** 2 * mask_delay
        ).sum(1)
        # sum and scale losses
        loss = NLL.mean() + kl_div.mean() + NVIL_term.mean() + baseline_loss.mean()
        ############################################################################
        # actual training step, i.e., backpropagation
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log losses
        self.log("loss/loss", loss, on_step=False, on_epoch=True)
        self.log("loss/NLL", NLL.mean(), on_step=False, on_epoch=True)
        self.log("loss/KL-Div-What", kl_div_what.sum(1).mean(), on_step=False, on_epoch=True)
        self.log("loss/KL-Div-Where", kl_div_where.sum(1).mean(), on_step=False, on_epoch=True)
        self.log("loss/KL-Div-Pres", kl_div_pres.sum(1).mean(), on_step=False, on_epoch=True)
        self.log("loss/baseline", baseline_loss.mean(), on_step=False, on_epoch=True)
        self.log("loss/NVIL_estimator", NVIL_term.mean(), on_step=False, on_epoch=True)
        # log count accuracy (labels are used here)
        batch_size = x.shape[0]
        true_counts = labels.view(batch_size, -1).max(1, keepdim=True)[0]
        acc = (results["counts"] == true_counts).sum().item() / batch_size
        self.log("metrics/count_accuracy", acc, on_step=False, on_epoch=True, logger=True)
        return {"loss": loss, "x": x}

    def training_epoch_end(self, outputs):
        """this function is called after each epoch"""
        step = self.current_epoch
        if (step + 1) % self.log_every_k_epochs == 0:
            n_samples = 7

            last_x = outputs[-1]["x"]
            i_samples = np.random.choice(range(len(last_x)), n_samples, False)
            images = last_x[i_samples]

            fig = self.plot_reconstructions_and_attention_rects(images)
            self.logger.experiment.add_figure("image and reconstructions", fig, global_step=step)
        return

    ########################################
    ######### TRAINING SETUP HOOKS #########
    ########################################

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.rnn.parameters()) + list(self.vae.parameters()),
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.baseline.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.base_weight_decay,
                },
            ]
        )
        return optimizer

    ########################################
    ########## DATA RELATED HOOKS ##########
    ########################################

    def prepare_data(self) -> None:
        if self.dataset_name == "MultiMNIST":
            self.dataset = MultiMNIST()
        return

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1000, num_workers=12, shuffle=True)