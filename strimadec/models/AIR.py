import torch
import pytorch_lightning as pl
import torch.distributions as dists
import torch.nn.functional as F

from strimadec.models.modules import VAE, RNN


class AIR(pl.LightningModule):

    """

        Attend-Infer-Repeat class as described by Eslami et al. (2016)

    Args:
        config (dict): dictionary containing the following configurations
            What VAE-Setup (dict): dictionary containing the setup for VAE of AIR
            RNN-Setup (dict): dictionary containing the setup of the RNN for z_pres, z_where
            RNN Baseline-Setup (dict): dictionary containing the setup for the RNN NVIL baseline

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
        self.lr, self.weight_decay = config["lr"], config["weight_decay"]
        self.log_every_k_epochs = config["log_every_k_epochs"]
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
                all_prob_pres (torch tensor): mean of z_pres for each N [batch, N]
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
            x_att_i = AIR.image_to_window(x, z_i_where, self.img_shape[0], self.window_dim)
            # put x_att_i through VAE
            x_tilde_att_i, z_i_what, mu_what_i, log_var_what_i = self.vae(x_att_i)
            # create image reconstruction
            x_tilde_i = AIR.window_to_image(x_tilde_att_i, z_i_where, self.img_shape)
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
                attention_rects[:, i] = AIR.get_attention_rectangle(
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
    ### MODEL SPECIFIC HELPER FUNCTIONS ####
    ########################################

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
                    "weight_decay": self.weight_decay,
                },
            ]
        )
        return optimizer