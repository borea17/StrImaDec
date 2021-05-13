import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.distributions as dists
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from strimadec.datasets.multi_object_dataset import MultiMNIST
from strimadec.models.modules import VAE, RNN
from strimadec.models.utils import gaussian_kl, bernoulli_kl


class DAIR(pl.LightningModule):

    """

        Discrete Attend-Infer-Repeat class using NVIL estimator for both the number of objects
        and the object class

    Args:
        config (dict): dictionary containing the following configurations
            What VAE-Setup (dict): dictionary containing the setup for VAE of AIR
            RNN-Setup (dict): dictionary containing the setup of the RNN for z_pres, z_where
            RNN Baseline-Setup (dict): dictionary containing the setup for the RNN NVIL baseline
            dataset_name (str): name of dataset
            number_of_slots_train (int): maximum number of objects assumed during training
            img_shape (list): shape of whole image provided as a list
            prior_z_pres (list): prior on z_pres
            prior_mean_z_where (list): prior on mean of z_where
            prior_var_z_where (list): prior on var of z_where
            lr (float): learning rate for VAE network parameters (ADAM)
            weight_decay (float): weight_decay for VAE network parameters (ADAM)

            base_lr (float): learning rate for baseline network parameters (ADAM)
            base_weight_decay (float): weight_decay for baseline network parameters (ADAM)
    """

    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    norm_target_rectangle = torch.tensor(
        [[-1.0, -1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    ).view(1, 3, 5)

    def __init__(self, config) -> None:
        super().__init__()
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
        prior_z_pres = torch.tensor(config["prior_z_pres"])
        prior_mean_z_where = torch.tensor(config["prior_mean_z_where"])
        prior_var_z_where = torch.tensor(config["prior_var_z_where"])
        self.register_buffer("prior_z_pres", prior_z_pres)
        self.register_buffer("prior_mean_z_where", prior_mean_z_where)
        self.register_buffer("prior_var_z_where", prior_var_z_where)
        # store some useful parameters as attributes
        self.window_dim = config["What VAE-Setup"]["img_dim"]
        self.z_what_dim = config["What VAE-Setup"]["latent_dim"]
        self.omega_dim = 1 + 2 * 3 + self.z_what_dim
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
            x_att_i = DAIR.image_to_window(x, z_i_where, self.img_shape[0], self.window_dim)
            # put x_att_i through VAE (get prototype)
            results_vae = self.vae(x_att_i)
            x_tilde_att_i = results_vae["x_tilde"]
            z_i_what = results_vae["z"]
            probs_logits_what_i = results_vae["probs_logits"]
            # create image reconstruction
            x_tilde_i = DAIR.window_to_image(x_tilde_att_i, z_i_where, self.img_shape)
            # update im1 with current versions
            z_im1 = torch.cat((z_i_pres, z_i_where, z_i_what), 1)
            h_im1 = h_i
            h_im1_b = h_i_b
            # put all distribution parameters into omega_i
            omega_i = torch.cat((prob_pres_i, mu_where_i, log_var_where_i, probs_logits_what_i), 1)
            # store intermediate results
            all_z[:, i : i + 1] = z_im1.unsqueeze(1)
            all_omega[:, i : i + 1] = omega_i.unsqueeze(1)
            all_x_tilde[:, i] = x_tilde_i
            baseline_values[:, i] = baseline_value
            # for nice visualization
            if save_attention_rectangle:
                attention_rects[:, i] = DAIR.get_attention_rectangle(
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
        p_z_pres = self.prior_z_pres.expand(q_z_pres.shape)  # [batch, N, 1]
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
        loss = NLL.mean() + 0.5 * kl_div.mean() + NVIL_term.mean() + baseline_loss.mean()
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
    ############# PLOT FUNCTIONS ###########
    ########################################

    def plot_reconstructions_and_attention_rects(self, images):
        """

        Args:
            images (torch.tensor) shape [batch_size, img_channels, img_dim, img_dim]
        """
        batch_size, img_channels = images.shape[0:2]
        colors_rect = ["red", "green", "yellow", "blue", "magenta"]
        # feed images through model
        results = self.forward(images.to(self.device), self.N_train, True)

        fig = plt.figure(figsize=(8, 3))
        for counter in range(batch_size):
            orig_img = images[counter]
            # plot original data
            ax = plt.subplot(2, batch_size, 1 + counter)
            if img_channels == 1:
                plt.imshow(orig_img[0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                plt.imshow(transforms.ToPILImage()(orig_img))
            plt.axis("off")
            if counter == 0:
                ax.annotate(
                    "Data",
                    xy=(-0.05, 0.5),
                    xycoords="axes fraction",
                    fontsize=14,
                    va="center",
                    ha="right",
                    rotation=90,
                )

            attention_rect = results["attention_rects"][counter]
            x_tilde = torch.clamp(results["x_tilde"][counter], 0, 1)
            # plot reconstruction
            ax = plt.subplot(2, batch_size, 1 + counter + batch_size)
            if img_channels == 1:
                plt.imshow(x_tilde[0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                plt.imshow(transforms.ToPILImage()(x_tilde))
            plt.axis("off")
            # show attention windows
            for step in range(self.N_train):
                rect = attention_rect[step].detach().cpu().numpy()
                if rect.sum() > 0:  # valid rectangle
                    plt.plot(rect[0], rect[1] - 0.5, color=colors_rect[step])
        return fig

    ########################################
    ### MODEL SPECIFIC HELPER FUNCTIONS ####
    ########################################

    @staticmethod
    def image_to_window(x, z_i_where, img_channels, window_size):
        grid_shape = (z_i_where.shape[0], img_channels, window_size, window_size)
        z_i_where_inv = AIR.invert_z_where(z_i_where)
        x_att_i = AIR.spatial_transform(x, z_i_where_inv, grid_shape)
        return x_att_i

    @staticmethod
    def window_to_image(x_tilde_att_i, z_i_where, img_shape):
        grid_shape = (z_i_where.shape[0], *img_shape)
        x_tilde_i = AIR.spatial_transform(x_tilde_att_i, z_i_where, grid_shape)
        return x_tilde_i

    @staticmethod
    def spatial_transform(x, z_where, grid_shape):
        theta_matrix = AIR.z_where_to_transformation_matrix(z_where)
        grid = F.affine_grid(theta_matrix, grid_shape, align_corners=False)
        out = F.grid_sample(x, grid, align_corners=False)
        return out

    @staticmethod
    def z_where_to_transformation_matrix(z_i_where):
        """taken from
        https://github.com/pyro-ppl/pyro/blob/dev/examples/air/air.py
        """
        batch_size = z_i_where.shape[0]
        out = torch.cat((z_i_where.new_zeros(batch_size, 1), z_i_where), 1)
        ix = AIR.expansion_indices
        if z_i_where.is_cuda:
            ix = ix.cuda()
        out = torch.index_select(out, 1, ix)
        theta_matrix = out.view(batch_size, 2, 3)
        return theta_matrix

    @staticmethod
    def invert_z_where(z_where):
        scale = z_where[:, 0:1] + 1e-9

        z_where_inv = torch.zeros_like(z_where)
        z_where_inv[:, 0:1] = 1 / scale
        z_where_inv[:, 1:3] = -z_where[:, 1:3] / scale
        return z_where_inv

    @staticmethod
    def get_attention_rectangle(z_i_where, img_size):
        """transforms a normalized target rectangle into a source rectangle using
        z_i_where and the image size to mimick image-to-window transformation

        Args:
            z_i_where (torch tensor): implicitely describing the transformation
            img_size (int): size of the whole image in one dimension

        Returns:
            source_rectangle (torch tensor): attented rectangle defined in image coordinates
        """
        batch_size = z_i_where.shape[0]
        z_i_where_inv = AIR.invert_z_where(z_i_where)
        theta_matrix = AIR.z_where_to_transformation_matrix(z_i_where_inv)
        target_rectangle = AIR.norm_target_rectangle.expand(batch_size, 3, 5).to(z_i_where.device)
        source_rectangle_normalized = torch.matmul(theta_matrix, target_rectangle)
        # remap into absolute values
        source_rectangle = 0 + (img_size / 2) * (source_rectangle_normalized + 1)
        return source_rectangle

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