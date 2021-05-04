import torch
import torch.nn as nn
import torch.distributions as dists


class VAE(nn.Module):
    """simple fully connected (FC) VAE class in which the encoder and decoder
    distribution needs to be specified,

    NOTE: this class can be used for inference, training/loss needs be formulated

    Args:
        config (dict): dictionary containing the following configurations
            img_channels (int): number of image channels
            img_dim (int): image dimension along one axis (assumed to be square)
            FC_hidden_dims_enc (list of ints): hidden dimensions of encoder
            FC_hidden_dims_dec (list of ints): hidden dimensions of decoder
            latent_dim (int): dimension of latent space
            encoder_distribution (str): encoding distribution ["Gaussian", "Categorical"]
            decoder_distribution (str): decoding distribution ["Gaussian", "Bernoulli"]
            ################## DEPENDENT ARGS ##################
            decoder_distribution == "GAUSSIAN":
                fixed_var (float): assumed variance of decoder distribution
    Attributes:
        encoder (nn.Sequential): encoder network
        decoder (nn.Sequential): decoder network
        encoder_distribution (str): encoding distribution ["Gaussian", "Categorical"]
        decoder_distribution (str): decoding distribution ["Gaussian", "Bernoulli"]
        img_dim (int): image dimension along one axis
        img_channels (int): number of color channels in image
        latent_dim (int): latent dimension of VAE
        ################## DEPENDENT ATTRIBUTES ##################
        decoder_distribution == "GAUSSIAN":
            fixed_var (float): assumed variance of decoder distribution
    """

    ENCODING_DISTS = ["Gaussian", "Categorical"]
    DECODING_DISTS = ["Gaussian", "Bernoulli"]

    def __init__(self, config) -> None:
        super().__init__()
        # parse config
        FC_hidden_dims_enc = config["FC_hidden_dims_enc"]
        FC_hidden_dims_dec = config["FC_hidden_dims_dec"]
        self.img_dim, self.img_channels = config["img_dim"], config["img_channels"]
        self.encoder_distribution = config["encoder_distribution"]
        self.decoder_distribution = config["decoder_distribution"]
        self.latent_dim = config["latent_dim"]
        assert (
            self.encoder_distribution in VAE.ENCODING_DISTS
        ), f"The encoding distribution is set to {self.encoder_distribution}, however only {VAE.ENCODING_DISTS} are supported."
        assert (
            self.decoder_distribution in VAE.DECODING_DISTS
        ), f"The decoding distribution is set to {self.decoder_distribution}, however only {VAE.DECODING_DISTS} are supported."
        # compute input size
        input_size = self.img_channels * (self.img_dim ** 2)
        # define output_dim of encoder
        if self.encoder_distribution == "Gaussian":
            output_dim_enc = 2 * self.latent_dim
        elif self.encoder_distribution == "Categorical":
            output_dim_enc = self.latent_dim
        # build encoder network
        FC_layers_encoder = nn.ModuleList(
            [nn.Flatten(), nn.Linear(input_size, FC_hidden_dims_enc[0]), nn.ReLU()]
        )
        for i in range(1, len(FC_hidden_dims_enc)):
            FC_layers_encoder.append(nn.Linear(FC_hidden_dims_enc[i - 1], FC_hidden_dims_enc[i]))
            FC_layers_encoder.append(nn.ReLU())
        FC_layers_encoder.append(nn.Linear(FC_hidden_dims_enc[-1], output_dim_enc))
        self.encoder = nn.Sequential(*FC_layers_encoder)
        # build decoder network
        FC_layers_decoder = nn.ModuleList(
            [nn.Linear(self.latent_dim, FC_hidden_dims_dec[0]), nn.ReLU()]
        )
        for i in range(1, len(FC_hidden_dims_dec)):
            FC_layers_decoder.append(nn.Linear(FC_hidden_dims_dec[i - 1], FC_hidden_dims_dec[i]))
            FC_layers_decoder.append(nn.ReLU())
        FC_layers_decoder.append(nn.Linear(FC_hidden_dims_dec[-1], input_size))
        if self.decoder_distribution == "Bernoulli":
            FC_layers_decoder.append(nn.Sigmoid())
        elif self.decoder_distribution == "Gaussian":
            self.fixed_var = config["fixed_var"]
        self.decoder = nn.Sequential(*FC_layers_decoder)
        # define example input shape
        self.example_input_shape = [64, self.img_channels, self.img_dim, self.img_dim]
        return

    def forward(self, x):
        results = {}
        alpha = self.encode(x)
        # get latent sample z depending on the distribution
        if self.encoder_distribution == "Gaussian":
            mu_E, log_var_E = torch.chunk(alpha, 2, dim=1)
            # sample noise variable for each batch
            epsilon = torch.randn_like(log_var_E)
            # get latent variable by reparametrization trick
            z = mu_E + torch.exp(0.5 * log_var_E) * epsilon
            results["mu_E"], results["log_var_E"], results["z"] = mu_E, log_var_E, z
        elif self.encoder_distribution == "Categorical":
            probs_logits = alpha
            # sample one-hot vector indices
            z_ind = dists.Categorical(logits=probs_logits).sample()
            # convert to one-hot vectors
            num_classes = probs_logits.shape[1]
            z = torch.nn.functional.one_hot(z_ind, num_classes=num_classes).type_as(probs_logits)
            results["probs_logits"], results["z"] = probs_logits, z
        # get reconstruction
        x_tilde = self.decode(z)
        results["x_tilde"] = x_tilde
        return results

    def encode(self, x):
        # get encoder distribution parameters
        alpha = self.encoder(x)
        return alpha

    def decode(self, z):
        # get decoder distribution parameters and reshape to [batch, img_channels, img_dim, img_dim]
        x_dec = self.decoder(z)
        if len(z.shape) == 2:
            x_tilde = x_dec.view(-1, self.img_channels, self.img_dim, self.img_dim)
        elif len(z.shape) == 3:  # this is only here for analytical loss computation
            batch_size = z.shape[0]
            x_tilde = x_dec.view(batch_size, -1, self.img_channels, self.img_dim, self.img_dim)
        if self.decoder_distribution == "Gaussian":
            # force output to be positive (EPS for numerical stability)
            pass
            # EPS = 1e-32
            # x_tilde = (x_tilde + EPS).abs()
        return x_tilde