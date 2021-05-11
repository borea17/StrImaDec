import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.autograd import grad

from strimadec.discrete_gradient_estimators.utils import set_requires_grad


def REBAR(probs_logits, target, temp, eta, model, loss_func):
    """
        computes the loss using the REBAR estimator (REINFORCE + CONCRETE)
        through which we can backpropagate, at the same time the temp hyperparameter
        is optimized to reduce the variance

    Args:
        probs_logits (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L] or [batch, img_channels, img_dim, img_dim]
        temp (tensor): temperature (tuneable) to use for the concrete distribution [1]
        eta (float): hyperparamer that scales the control variate
        model (nn.Module): model parameters that should not be kept fixed when optimizing variance
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        estimator (tensor): batch-wise loss (including variance loss) [batch]
    """
    # compute log probabilities and probabilities [batch, L] (use EPS for numerical stability)
    EPS = 1e-16
    probs = F.softmax(probs_logits, dim=1).clamp(min=EPS, max=1 - EPS)
    log_probs = probs.log()
    # sample unit noise u, v (exclude 0, since log won't work otherwise) [batch, L]
    u = torch.FloatTensor(*log_probs.shape).uniform_(1e-38, 1.0).to(probs_logits.device)
    v = torch.FloatTensor(*log_probs.shape).uniform_(1e-38, 1.0).to(probs_logits.device)
    # convert u to u_Gumbel [batch, L]
    u_Gumbel = -torch.log(-torch.log(u))
    # Gumbel Max Trick to obtain discrete latent p(z|x) [batch, L]
    z_ind = (log_probs + u_Gumbel).argmax(1)
    z = F.one_hot(z_ind, num_classes=log_probs.shape[1]).type_as(log_probs).detach()
    # Gumbel Softmax to obtain relaxed latent p(z_tilde|x) [batch, L]
    z_tilde = torch.softmax((log_probs + u_Gumbel) / temp, dim=1)
    # sample s_tilde from p(s_tilde|z), see appendix D of Tucker et al. 2017
    v_Gumbel = -torch.log(-torch.log(v))  # b =1
    v_nonGumbel = -torch.log(-(v.log() / probs) - v.log())  # otherwise
    v_prime = (1.0 - z) * v_nonGumbel + z * v_Gumbel
    s_tilde = torch.softmax(v_prime / temp, dim=1)
    # compute REBAR estimator (evaluate loss_func at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)  # [batch]
    f_z_tilde = loss_func(z_tilde, target)  # [batch]
    f_s_tilde = loss_func(s_tilde, target)  # [batch]
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind)  # [batch]
    # compute gradient estimator (temp is set fixed in backward and won't be affected)
    estimator = (f_z - eta * f_s_tilde).detach() * log_prob + eta * (f_z_tilde - f_s_tilde)
    # compute variance estimator (use partial derivatives for the sake of clarity)
    g_log_prob = grad(
        log_prob,
        probs_logits,
        grad_outputs=torch.ones_like(log_prob),
        create_graph=True,
        retain_graph=True,
    )[0]
    g_f_z_tilde = grad(
        f_z_tilde,
        probs_logits,
        grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True,
        retain_graph=True,
    )[0]
    g_f_s_tilde = grad(
        f_s_tilde,
        probs_logits,
        grad_outputs=torch.ones_like(f_s_tilde),
        create_graph=True,
        retain_graph=True,
    )[0]
    # compute gradient estimator as a function of temp [batch, L]
    g_estimator = (f_z - eta * f_s_tilde).unsqueeze(1) * g_log_prob + eta * (
        g_f_z_tilde - g_f_s_tilde
    )
    if type(model).__name__ == "DVAEST_LossModel":  # avoid not implemented error of Hessian
        # avoid RuntimeError: derivative for grid_sampler_2d_backward is not implemented
        g_estimator = (f_z - eta * f_s_tilde).unsqueeze(1) * g_log_prob + eta * (
            g_f_z_tilde - g_f_s_tilde
        ).detach()
    # compute variance estimator [batch, L]
    var_estimator = g_estimator ** 2
    # obtain only gradients for temp (fix model parameters)
    set_requires_grad(model, False)
    var_estimator.sum(1).mean().backward(retain_graph=True)
    set_requires_grad(model, True)
    return estimator, f_z
