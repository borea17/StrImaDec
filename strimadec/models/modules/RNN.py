import torch
import torch.nn as nn


class RNN(nn.Module):

    """simple  fully connected (FC) RNN class only using linear leayers for use in AIR

    Args:
        config (dict): dictionary containing the following configurations
            img_channels (int): number of image channels
            img_dim (int): image dimension along one axis (asummed to be square)
            hidden_state_dim (int): size of hidden state in MLP-RNN (input to MLP)
            latent_space_dim (int): size of whole latent space (input to MLP)
            FC_hidden_dims (list of ints): hidden dims of RNN
            output_size (int): size of (visible) output (distribution params)
                whole_out_put = output_size + hidden_state_dim
            output_bias_init (list): initialization of bias in output layer
            baseline_net (bool): boolean indicator whether RNN is used as a NVIL baseline
    """

    def __init__(self, config):
        super(RNN, self).__init__()
        img_channels = config["img_channels"]
        img_dim = config["img_dim"]
        hidden_state_dim = config["hidden_state_dim"]
        latent_space_dim = config["latent_space_dim"]
        FC_hidden_dims = config["FC_hidden_dims"]
        output_size = config["output_size"]
        output_bias_init = config["output_bias_init"]

        input_size = (img_dim ** 2) * img_channels + hidden_state_dim + latent_space_dim
        # define FC_RNN layers
        FC_RNN_layers = nn.ModuleList([nn.Linear(input_size, FC_hidden_dims[0]), nn.ReLU()])
        for i in range(1, len(FC_hidden_dims)):
            FC_RNN_layers.append(nn.Linear(FC_hidden_dims[i - 1], FC_hidden_dims[i]))
            FC_RNN_layers.append(nn.ReLU())
        # output layer configuration
        output_layer = nn.Linear(FC_hidden_dims[-1], output_size + hidden_state_dim)
        # initialize output layer
        output_layer.weight.data[0:output_size] = nn.Parameter(
            torch.zeros(output_size, FC_hidden_dims[-1])
        )
        output_layer.bias.data[0:output_size] = nn.Parameter(torch.tensor(output_bias_init))
        FC_RNN_layers.append(output_layer)
        self.fc_rnn = nn.Sequential(*FC_RNN_layers)
        self.output_size = output_size
        self.baseline_net = config["baseline_net"]
        return

    def forward(self, x, z_im1, h_im1):
        batch_size = x.shape[0]
        rnn_input = torch.cat((x.view(batch_size, -1), z_im1, h_im1), dim=1)
        rnn_output = self.fc_rnn(rnn_input)
        omega_i = rnn_output[:, 0 : self.output_size]
        h_i = rnn_output[:, self.output_size : :]
        if not self.baseline_net:
            # omega_i[:, 0] corresponds to z_pres probability
            omega_i[:, 0] = torch.sigmoid(omega_i[:, 0])
        return omega_i, h_i