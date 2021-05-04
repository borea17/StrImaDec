import torch.nn as nn


class DVAE_LossModel(nn.Module):
    def __init__(self, discrete_vae):
        super().__init__()
        self.VAE = discrete_vae
        return

    def loss_func(self, z, x):
        batch_size, L = z.shape[0], z.shape[1]
        x_tilde = self.VAE.decode(z)

        if self.VAE.decoder_distribution == "Gaussian":
            # NLL corresponds to scaled MSE loss
            NLL = -((1 / (2 * self.VAE.fixed_var)) * (x - x_tilde) ** 2).view(batch_size, L, -1)
        elif self.VAE.decoder_distribution == "Bernoulli":
            # NLL corresponds to binary cross entropy loss
            NLL = -(x * x_tilde.log() + (1 - x) * (1 - x_tilde).log()).view(batch_size, L, -1)
            # NOTE: if x is not binary, this is a not a valid NLL (however, it still works)
        if len(x.shape) < 5:  # analytical estimator needs [batch, L, -1]
            # NLL shape [batch, -1]
            NLL = NLL.view(batch_size, -1)
        return NLL