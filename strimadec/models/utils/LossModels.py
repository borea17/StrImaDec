import torch.nn as nn

from strimadec.models.utils.kl_divergences import gaussian_kl


class DVAE_LossModel(nn.Module):
    def __init__(self, discrete_vae):
        super().__init__()
        self.VAE = discrete_vae
        return

    def loss_func(self, z, x):
        x_tilde = self.VAE.decode(z)
        # compute batch-wise negative log-likelihood by summing over image dimensions
        if self.VAE.decoder_distribution == "Gaussian":
            # NLL corresponds to scaled MSE loss
            NLL = ((1 / (2 * self.VAE.fixed_var)) * (x - x_tilde) ** 2).sum(axis=(-1, -2, -3))
        elif self.VAE.decoder_distribution == "Bernoulli":
            # NLL corresponds to binary cross entropy loss
            NLL = -(x * x_tilde.log() + (1 - x) * (1 - x_tilde).log()).sum(axis=(-1, -2, -3))
            # NOTE: if x is not binary, this is a not a valid NLL (however, it still works)
        return NLL


class DVAEST_LossModel(nn.Module):
    def __init__(self, discrete_vae, localization_net):
        super().__init__()
        self.VAE = discrete_vae
        self.localization_net = localization_net
        return

    def compute_kl_div(self, results_localization_net):
        mu_transform = results_localization_net["mu_t"]  # [batch, 6]
        log_var_transform = results_localization_net["log_var_t"]  # [batch, 6]
        prior_mu_transform = self.localization_net.prior_mu_transform  # [1, 6]
        prior_var_transform = self.localization_net.prior_var_transform  # [1, 6]
        # compute kl divergence between transform_dist and prior_transform_dist [batch_size]
        q_transform_dist = [mu_transform, log_var_transform.exp()]
        p_transform_dist = [prior_mu_transform, prior_var_transform]
        kl_div = gaussian_kl(q_transform_dist, p_transform_dist)
        return kl_div

    def retrieve_kl_div_position(self):
        return self.kl_div_position

    def loss_func(self, z, x):
        # compute prototype x_p [batch, img_channels, img_dim, img_dim]
        x_p = self.VAE.decode(z)
        # infer transformation distribution parameters and sample from distribution
        results_localization_net = self.localization_net.forward(x)
        # compute x_tilde by transforming x_p with t
        x_tilde = self.localization_net.spatial_transform(x_p, results_localization_net["t"])
        # compute batch-wise negative log-likelihood by summing over image dimensions
        if self.VAE.decoder_distribution == "Gaussian":
            # NLL corresponds to scaled MSE loss
            NLL = ((1 / (2 * self.VAE.fixed_var)) * (x - x_tilde) ** 2).sum(axis=(-1, -2, -3))
        elif self.VAE.decoder_distribution == "Bernoulli":
            # NLL corresponds to binary cross entropy loss
            NLL = -(x * x_tilde.log() + (1 - x) * (1 - x_tilde).log()).sum(axis=(-1, -2, -3))
            # NOTE: if x is not binary, this is a not a valid NLL (however, it still works)
        # assign kl_div as attribute instead of returning it to have a common interface
        self.kl_div_position = self.compute_kl_div(results_localization_net)
        return NLL
