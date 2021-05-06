import torch
import torch.nn as nn
import torch.distributions as dists


class BaselineNet(nn.Module):

    """simple fully connected (FC) neural baseline for use in NVIL or RELAX estimator

    Args:
        config (dict): dictionary containing the following configurations
            input_dim (int): input dimension of FC network
            FC_hidden_dims (list of ints): hidden dimensions of baseline network
            output_dim (int): output dimension: 1 (NVIL) or num_classes (RELAX)
            ################### OPTIONAL PARAMS ###################
            log_temp_init (float): init temperature of concrete distribution
    """

    def __init__(self, config):
        super().__init__()
        # parse input
        FC_hidden_dims = config["FC_hidden_dims"]
        input_dim, output_dim = config["input_dim"], config["output_dim"]
        # define baseline network based on FC_hidden_dims
        layers = nn.ModuleList([nn.Flatten(), nn.Linear(input_dim, FC_hidden_dims[0]), nn.ReLU()])
        for i in range(1, len(FC_hidden_dims)):
            layers.append(nn.Linear(FC_hidden_dims[i - 1], FC_hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(FC_hidden_dims[-1], output_dim))
        self.baseline = nn.Sequential(*layers)
        if "log_temp_init" in config:
            log_temp_init = config["log_temp_init"]
            self.log_temp = torch.nn.Parameter(torch.tensor(log_temp_init), requires_grad=True)
            self.contains_temp = True
        else:
            self.contains_temp = False
        return

    def forward(self, x):
        if self.contains_temp:
            temp = self.log_temp.exp()
            x = torch.softmax(x / temp, dim=1)
        out = self.baseline(x)
        return out
