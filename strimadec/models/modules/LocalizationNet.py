import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationNet(nn.Module):

    """

        simple fully connected (FC) localization network that is used to estimate
        the distribution (Gaussian) parameters (mean and var) of the Spatial Transformer (ST)

    Args:
        config (dict): dictionary containing the following configurations
            input_dim (int): input dimension of FC network
            FC_hidden_dims (list of ints): hidden dimensions of localization network
            prior_mu_transform (list): initialization for mean [output_dim]
            prior_var_transform (list): initialization for var [output_dim]
            ################### OPTIONAL PARAMS ###################
            log_temp_init (float): init temperature of concrete distribution
    """

    def __init__(self, config):
        super().__init__()
        OUTPUT_DIM = 6  # fixed output dim for mean/log_var
        # parse input
        FC_hidden_dims = config["FC_hidden_dims"]
        input_dim = config["input_dim"]
        prior_mu_transform = torch.tensor(config["prior_mu_transform"])
        prior_var_transform = torch.tensor(config["prior_var_transform"])
        # define localization network based on FC_hidden_dims
        localization_layers = nn.ModuleList(
            [nn.Flatten(), nn.Linear(input_dim, FC_hidden_dims[0]), nn.ReLU()]
        )
        for i in range(1, len(FC_hidden_dims)):
            localization_layers.append(nn.Linear(FC_hidden_dims[i - 1], FC_hidden_dims[i]))
            localization_layers.append(nn.ReLU())
        # define last layer and initialize it using prior_mu and prior_var
        loc_regression_layer = nn.Linear(FC_hidden_dims[-1], 2 * OUTPUT_DIM)
        loc_regression_layer.weight.data.fill_(0)
        loc_regression_layer.bias.data[0:OUTPUT_DIM] = prior_mu_transform
        loc_regression_layer.bias.data[OUTPUT_DIM::] = prior_var_transform.log()
        # append last layer to localization
        localization_layers.append(loc_regression_layer)
        self.localization_net = nn.Sequential(*localization_layers)
        # register prior_mu_transform and prior_var_transform
        self.register_buffer("prior_mu_transform", prior_mu_transform.view(1, 6))
        self.register_buffer("prior_var_transform", prior_var_transform.view(1, 6))
        return

    def forward(self, x):
        """
            infer distribution parameters of transformation from x and sample transformation
            vector t

        Args:
            x (torch tensor): original image [batch, img_channels, img_dim, img_dim]

        Returns:
            results (dict): results dictionary containing
                t (torch tensor): sampled transformation parameter t [batch, 6]
                mu_t (torch tensor): mean of sampled transformation t [batch, 6]
                log_var_t (torch tensor): log variance of sampled transformation t [batch, 6]
        """
        # get affine transformation parameters [batch, 6]
        alpha = self.localization_net(x)
        mu_transform, log_var_transform = torch.chunk(alpha, 2, dim=1)
        # sample from affine distribution to obtain parameter vector t (parameterization trick)
        epsilon = torch.randn_like(log_var_transform)
        t = mu_transform + torch.exp(0.5 * log_var_transform) * epsilon
        results = {"mu_t": mu_transform, "log_var_t": log_var_transform, "t": t}
        return results

    def spatial_transform(self, x_p, t):
        """
            spatially transform x_p based on sampled transformation parameter t

        Args:
            x_p (torch tensor): prototype image [batch, img_channels, img_dim, img_dim]
            t (torch tensor): sampled transformation parameter vector t [batch, 6]
        """
        transformation_matrix = t.view(-1, 2, 3)
        # grid generator
        if len(x_p.shape) == 5:  # analytical gradient estimator
            # [batch, L, img_channels, img_dim, img_dim]->[batch*L, img_channels, img_dim, img_dim]
            num_classes = x_p.shape[1]  # L
            transformation_matrix = transformation_matrix.repeat(num_classes, 1, 1)
            x_p = torch.cat(x_p.chunk(num_classes, 1), 0).squeeze(1)
            grid = F.affine_grid(transformation_matrix, x_p.size(), align_corners=False)
            x_tilde = F.grid_sample(x_p, grid, align_corners=False)
            # [batch*L, img_channels, img_dim, img_dim]->[batch, L, img_channels, img_dim, img_dim]
            x_tilde = torch.stack(x_tilde.chunk(num_classes, dim=0), 1)
            return x_tilde
        grid = F.affine_grid(transformation_matrix, x_p.size(), align_corners=False)
        # sampler [batch, img_channels, img_dim, img_dim]
        x_tilde = F.grid_sample(x_p, grid, align_corners=False)
        return x_tilde
