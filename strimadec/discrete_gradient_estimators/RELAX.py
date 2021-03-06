import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.autograd import grad

from strimadec.discrete_gradient_estimators.utils import set_requires_grad


def RELAX(probs_logits, target, c_phi, model, loss_func):
    """
        computes the loss using the RELAX estimator (REBAR + NVIL)
        through which we can backpropagate, at the same time the network parameters
        of c_phi are optimized to reduce the variance

    Args:
        probs (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L]
        c_phi (nn.Sequential): neural control variate for RELAX
        model (nn.Module): model parameters that should not be kept fixed when optimizing variance
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        estimator (tensor): batch-wise loss (including variance loss) [batch]
    """
    # compute log probabilities and probabilities [batch, L] (use EPS for numerical stability)
    EPS = 1e-16
    probs = F.softmax(probs_logits, dim=1).clamp(min=EPS, max=1 - EPS)
    log_probs = probs.log()
    # sample unit noise u, v (exclude 0, since log won't work otherwise)
    u = torch.FloatTensor(*log_probs.shape).uniform_(1e-38, 1.0).to(probs_logits.device)
    v = torch.FloatTensor(*log_probs.shape).uniform_(1e-38, 1.0).to(probs_logits.device)
    # convert u to u_Gumbel
    u_Gumbel = -torch.log(-torch.log(u))
    # Gumbel Max Trick to obtain discrete latent p(z|x)
    z_ind = (log_probs + u_Gumbel).argmax(1)
    z = F.one_hot(z_ind, num_classes=log_probs.shape[1]).type_as(log_probs).detach()
    # generate z_tilde
    z_tilde = c_phi(log_probs + u_Gumbel)
    # sample s_tilde from p(s_tilde|z), see appendix D of Tucker et al. 2017
    v_b = v[torch.arange(probs.shape[0]), z_ind].unsqueeze(1)
    v_Gumbel = -torch.log(-torch.log(v))
    v_prime = -torch.log(-(torch.log(v) / probs) - torch.log(v_b))
    v_prime[torch.arange(probs.shape[0]), z_ind] = v_Gumbel[torch.arange(probs.shape[0]), z_ind]
    s_tilde = c_phi(v_prime)
    # compute RELAX estimator (evaluate loss at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)  # [batch]
    f_z_tilde = loss_func(z_tilde, target)  # [batch]
    f_s_tilde = loss_func(s_tilde, target)  # [batch]
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind)  # [batch]
    # compute gradient estimator (c_phi is set fixed in backward and won't be affected)
    estimator = (f_z - f_s_tilde).detach() * log_prob + f_z_tilde - f_s_tilde
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
    # compute gradient estimator as a function of eta and temp [batch, L]
    g_estimator = (f_z - f_s_tilde).unsqueeze(1) * g_log_prob + (g_f_z_tilde - g_f_s_tilde)
    if type(model).__name__ == "DVAEST_LossModel":  # avoid not implemented error of Hessian
        # avoid RuntimeError: derivative for grid_sampler_2d_backward is not implemented
        g_estimator = (f_z - f_s_tilde).unsqueeze(1) * g_log_prob + (
            g_f_z_tilde - g_f_s_tilde
        ).detach()
    # compute variance estimator [batch, L]
    var_estimator = g_estimator ** 2
    # obtain only gradients for c_phi (fix model parameters)
    set_requires_grad(model, False)
    var_estimator.sum(1).mean().backward(retain_graph=True)
    set_requires_grad(model, True)
    return estimator, f_z
