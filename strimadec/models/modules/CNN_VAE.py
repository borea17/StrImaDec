import torch
import torch.nn as nn
import numpy as np


class CNN_VAE(nn.Module):
    """simple CNN-VAE class with a Gaussian encoder (mean and diagonal variance
    structure) and a Gaussian decoder with fixed variance for use in AIR

    with a Gaussian encoder (mean and diagonal variance
    structure) and a Gaussian decoder with fixed variance for use in AIR

    Args:
        config (dict): dictionary containing the following configurations
            in_channels (int): number of input channels
            img_dim (int): image dimension along one axis (assumed to be square)
            num_conv_layers_enc (int): number of convolutional layers in encoder
            base_channel_enc (int): base number of channels in encoder (multiplied by some number)
            max_channel_multiplier_enc (int): upper-bound multiplier of base_channel_enc
            MLP_hidden_dims_enc (list of ints): hidden dimensions of encoder MLP
            latent_dim (int): dimension of latent space
            num_conv_layers_dec (int): number of convolutional layers in decoder
            base_channel_dec (int): base number of channels in decoder (multiplied by some number)
            max_channel_multiplier_dec (int): upper-bound multiplier of base_channel_dec


    Attributes:
        encoder (nn.Sequential): encoder network for mean and log_var
        decoder (nn.Sequential): spatial broadcast decoder network for mean (fixed var)
        img_dim (int): image dimension along one axis
        expand_dim (int): expansion of latent image to accomodate for lack of padding
        x_grid (torch tensor): appended x coordinates for spatial broadcast decoder
        y_grid (torch tensor): appended x coordinates for spatial broadcast decoder
    """

    def __init__(self, config):
        super(CNN_VAE, self).__init__()
        in_channels = config["in_channels"]
        img_dim = config["img_dim"]
        self.img_dim = img_dim  # need in decode
        # CNN encoder configuration
        num_conv_layers_enc = config["num_conv_layers_enc"]
        base_channel_enc = config["base_channel_enc"]
        max_channel_multiplier_enc = config["max_channel_multiplier_enc"]
        # define convolutional encoder network
        conv_layers_encoder = nn.ModuleList(
            [
                nn.Conv2d(in_channels, base_channel_enc, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]
        )
        last_out_ch = base_channel_enc
        for i in range(num_conv_layers_enc - 1):
            current_in_ch = last_out_ch
            current_out_ch = base_channel_enc * min((i + 1), max_channel_multiplier_enc)

            conv_layers_encoder.append(
                nn.Conv2d(current_in_ch, current_out_ch, kernel_size=3, stride=2, padding=1)
            )
            conv_layers_encoder.append(nn.ReLU())

            last_out_ch = current_out_ch
        # MLP encoder configuration
        assert img_dim / 2 ** (num_conv_layers_enc) == int(img_dim / 2 ** (num_conv_layers_enc))
        input_dimension = (int(img_dim / 2 ** (num_conv_layers_enc)) ** 2) * last_out_ch
        hidden_dims = config["MLP_hidden_dims_enc"]
        latent_dim = config["latent_dim"]
        # define MLP encoder
        MLP_layers_encoder = nn.ModuleList(
            [nn.Flatten(), nn.Linear(input_dimension, hidden_dims[0]), nn.ReLU()]
        )
        for i in range(1, len(hidden_dims)):
            MLP_layers_encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            MLP_layers_encoder.append(nn.ReLU())
        MLP_layers_encoder.append(nn.Linear(hidden_dims[-1], 2 * latent_dim))
        # put conv layer and MLP together in sequential encoder
        self.encoder = nn.Sequential(*conv_layers_encoder.extend(MLP_layers_encoder))
        # spatial broadcast decoder configuration
        num_conv_layers_dec = config["num_conv_layers_dec"]
        base_channel_dec = config["base_channel_dec"]
        max_channel_multiplier_dec = config["max_channel_multiplier_dec"]
        # expand_dim is needed to accommodate for the lack of padding
        self.expand_dim = 2 * num_conv_layers_dec
        # define x_grid and y_grid (needed for SBD)
        x = torch.linspace(-1, 1, img_dim + self.expand_dim)
        y = torch.linspace(-1, 1, img_dim + self.expand_dim)
        x_grid, y_grid = torch.meshgrid(x, y)
        # reshape into [1, 1, img_dim, img_dim] and save in state_dict
        self.register_buffer("x_grid", x_grid.view((1, 1) + x_grid.shape).clone())
        self.register_buffer("y_grid", y_grid.view((1, 1) + y_grid.shape).clone())
        conv_layers_dec = nn.ModuleList(
            [
                nn.Conv2d(latent_dim + 2, base_channel_dec, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            ]
        )
        last_out_ch = base_channel_dec
        for i in range(num_conv_layers_dec - 1):
            current_in_ch = last_out_ch
            current_out_ch = base_channel_dec * min((i + 1), max_channel_multiplier_dec)

            conv_layers_dec.append(
                nn.Conv2d(current_in_ch, current_out_ch, kernel_size=3, stride=1, padding=0)
            )
            conv_layers_dec.append(nn.ReLU())

            last_out_ch = current_out_ch
        conv_layers_dec.append(
            nn.Conv2d(last_out_ch, in_channels, kernel_size=1, stride=1, padding=0)
        )
        # put conv layer in sequential decoder
        self.decoder = nn.Sequential(*conv_layers_dec)
        return

    def forward(self, x):
        [z, mu_E, log_var_E] = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z, mu_E, log_var_E

    def encode(self, x):
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # sample noise variable for each batch
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5 * log_var_E) * epsilon
        return [z, mu_E, log_var_E]

    def decode(self, z):
        batch_size = z.shape[0]
        # reshape z into [batch_size, latent_dim, 1, 1]
        z = z.view(z.shape + (1, 1))
        # tile across image [batch_size, latent_im, img_dim + expand_dim, img_dim + expand_dim]
        z_b = z.repeat(1, 1, self.img_dim + self.expand_dim, self.img_dim + self.expand_dim)
        # upsample x_grid and y_grid to [batch_size, 1, img_dim + expand_dim, img_dim + expand_dim]
        x_b = self.x_grid.repeat(batch_size, 1, 1, 1)
        y_b = self.y_grid.repeat(batch_size, 1, 1, 1)
        # concatenate vectors [batch_size, latent_dim + 2, img_dim + expand_dim, img_dim + expand_dim]
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        # apply convolutional layers mu_D
        mu_D = self.decoder(z_sb)
        return mu_D