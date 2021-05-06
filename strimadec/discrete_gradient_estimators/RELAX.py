import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.autograd import grad


def RELAX(probs_logits, target, c_phi, loss_func):
    """
        computes the loss using the RELAX estimator (REBAR + NVIL)
        through which we can backpropagate, at the same time the network parameters
        of c_phi are optimized to reduce the variance

    Args:
        probs (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L]
        c_phi (nn.Sequential): neural control variate for RELAX
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        estimator (tensor): batch-wise loss (including variance loss) [batch]
    """
    # compute log probabilities and probabilities
    log_probs = F.log_softmax(probs_logits, dim=1)
    probs = F.softmax(probs_logits, dim=1)
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

    # v_Gumbel = -torch.log(-torch.log(v))  # b =1
    # v_nonGumbel = -torch.log(-(v.log() / probs) - v.log())  # otherwise
    # v_prime = (1.-z)*v_nonGumbel + z*v_Gumbel

    v_Gumbel = -torch.log(-torch.log(v))  # b =1
    topgumbels = v_Gumbel + torch.logsumexp(log_probs, axis=1, keepdims=True)
    topgumbel = torch.sum(z*topgumbels, axis=-1, keepdims=True)
    def truncated_gumbel(gumbel, truncation):
        EPSILON = 1e-16 
        return -torch.log(EPSILON + torch.exp(-gumbel) + torch.exp(-truncation))
    truncgumbel = truncated_gumbel(v_Gumbel + log_probs, topgumbel)

    v_prime = (1. - z)*truncgumbel + z*topgumbels
    s_tilde = c_phi(v_prime)
    # compute RELAX estimator (evaluate loss at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)  # [batch]
    f_z_tilde = loss_func(z_tilde, target)  # [batch]
    f_s_tilde = loss_func(s_tilde, target)  # [batch]
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind)  # [batch]
    # compute gradient estimator (detach c_phi such that backward won't affect it)
    estimator = (f_z - f_s_tilde).detach() * log_prob + f_z_tilde.detach() - f_s_tilde.detach()
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
    # compute variance estimator [batch, L]
    var_estimator = g_estimator ** 2
    return estimator + var_estimator.sum(1) - var_estimator.detach().sum(1), f_z
