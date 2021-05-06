import torch


def loss_func(x, y):
    num_classes = y.shape[-1]
    return (1 / num_classes) * ((x - y) ** 2).sum(axis=-1)


def build_experimental_setup(estimator_name, target, num_epochs, SEED):
    """
        creates the experimental setup params given the estimator_name

    Args:
        estimator_name (str): name of gradient estimator
        target (torch tensor): target tensor [1, num_classes]
        num_epochs (int): number of epochs to train
        SEED (int): seed that shall be used for this experiment

    Returns:
        params (dict): dictionary that can be used to execute
            `run_stochastic_optimization`
    """
    x = torch.ones([1, 1])
    # use the simplest possible network (with weights initialized to zero)
    num_classes = target.shape[1]
    linear_layer_encoder = torch.nn.Linear(1, num_classes, bias=False)
    linear_layer_encoder.weight.data.fill_(0.0)
    encoder_net = torch.nn.Sequential(linear_layer_encoder)
    # define independent params dictionary
    params = {
        "SEED": SEED,
        "x": x,
        "target": target,
        "encoder_net": encoder_net,
        "num_epochs": num_epochs,
        "batch_size": 1,
        "lr": 0.01,
        # "loss_func": lambda x, y: (1 / num_classes) * (x - y) ** 2,
        "loss_func": loss_func,
        "estimator_name": estimator_name,
        "FIXED_BATCH": 1000,
    }
    # extend dict by estimator dependent params
    if estimator_name == "NVIL":
        torch.manual_seed(SEED)  # seed here to make network initializations deterministic
        baseline_net = torch.nn.Sequential(torch.nn.Linear(1, 1))
        params["baseline_net"] = baseline_net
        params["tune_lr"] = 0.1
    elif estimator_name == "CONCRETE":
        params["temp"] = 0.6
    elif estimator_name == "REBAR":
        params["eta"] = 1.0
        params["log_temp"] = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        params["tune_lr"] = 0.001
    elif estimator_name == "RELAX":

        class C_PHI(torch.nn.Module):
            """
                Control variate for RELAX,
                NOTE: this is only the neural part of the control variate which will be used as
                      the adapted input in the loss function

            Args:
                num_classes (int): number of classes
                log_temp_init (float): logarithmized init temperature for continuous relaxation
            """

            def __init__(self, num_classes, log_temp_init):
                super(C_PHI, self).__init__()
                self.network = torch.nn.Sequential(torch.nn.Linear(num_classes, num_classes))
                self.log_temp = torch.nn.Parameter(torch.tensor(log_temp_init), requires_grad=True)
                return

            def forward(self, z):
                temp = self.log_temp.exp()
                z_tilde = torch.softmax(z / temp, dim=1)
                out = self.network(z_tilde)
                return out

        torch.manual_seed(SEED)  # seed here to make network initializations deterministic
        params["tune_lr"] = 0.01
        params["c_phi"] = C_PHI(num_classes=num_classes, log_temp_init=0.5)
    return params