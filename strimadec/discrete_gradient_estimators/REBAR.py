import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.autograd import grad


def REBAR(probs_logits, target, temp, eta, loss_func):
    """
        computes the loss using the REBAR estimator (REINFORCE + CONCRETE)
        through which we can backpropagate, at the same time the hyperparameters
        temp and eta are optimized to reduce the variance

    Args:
        probs_logits (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L] or [batch, img_channels, img_dim, img_dim]
        temp (tensor): temperature (tuneable) to use for the concrete distribution [1]
        eta (float): hyperparamer that scales the control variate
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        estimator (tensor): batch-wise loss (including variance loss) [batch]
    """
    # compute log probabilities and probabilities [batch, L] (use EPS for numerical stability)
    EPS = 1e-9
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
    v_prime = (1. - z) * v_nonGumbel + z * v_Gumbel
    s_tilde = torch.softmax(v_prime / temp, dim=1)
    # compute REBAR estimator (evaluate loss_func at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)  # [batch]
    f_z_tilde = loss_func(z_tilde, target)  # [batch]
    f_s_tilde = loss_func(s_tilde, target)  # [batch]
    z_tilde_detach_temp = torch.softmax((log_probs + u_Gumbel) / temp.detach(), dim=1)
    s_tilde_detach_temp = torch.softmax(v_prime / temp.detach(), dim=1)
    f_z_tilde_detach_temp = loss_func(z_tilde_detach_temp, target)
    f_s_tilde_detach_temp = loss_func(s_tilde_detach_temp, target)
    z_tilde_detach_probs = torch.softmax((log_probs.detach() + u_Gumbel) / temp, dim=1)
    s_tilde_detach_probs = torch.softmax(v_prime.detach() / temp, dim=1)
    f_z_tilde_detach_probs = loss_func(z_tilde_detach_probs, target)
    f_s_tilde_detach_probs = loss_func(s_tilde_detach_probs, target)
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind)  # [batch]
    # compute gradient estimator (detach temp such that backward won't affect it)
    estimator = (f_z - eta * f_s_tilde).detach() * log_prob + eta * (
        f_z_tilde_detach_temp - f_s_tilde_detach_temp
    )
    # estimator = (f_z - eta * f_s_tilde).detach() * log_prob + eta * (f_z_tilde - f_s_tilde)
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
    g_estimator = (
        (f_z - eta * f_s_tilde_detach_probs).unsqueeze(1) * g_log_prob.detach() + eta * (g_f_z_tilde - g_f_s_tilde)
    )
    # compute variance estimator [batch, L] (use EPS for numerical stability)
    var_estimator = (g_estimator ** 2).clamp(min=EPS)
    # return estimator + var_estimator.sum(1), f_z
    # return estimator + var_estimator.sum(1) - var_estimator.detach().sum(1), f_z
    return estimator, f_z
