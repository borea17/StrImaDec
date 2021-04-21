import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.autograd import grad


def RELAX(probs_logits, target, c_phi, loss_func):
    """
        computes the loss using the RELAX estimator (REINFORCE + CONCRETE + NVIL)
        through which we can backpropagate, at the same time the hyperparameters
        temp and eta are optimized to reduce the variance

    Args:
        probs (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L]
        c_phi (nn.Sequential): neural control variate for RELAX
        loss_func (method): loss function that takes the sampled class vectors as input
    """
    # compute log probabilities and probabilities
    log_probs = F.log_softmax(probs_logits, dim=1)
    probs = F.softmax(probs_logits, dim=1)
    # sample unit noise u, v
    u = torch.rand(log_probs.shape).to(probs_logits.device)
    v = torch.rand(log_probs.shape).to(probs_logits.device)
    # convert u to u_Gumbel
    u_Gumbel = -torch.log(-torch.log(u))
    # Gumbel Max Trick to obtain discrete latent p(z|x)
    z_ind = (log_probs + u_Gumbel).argmax(1)
    z = F.one_hot(z_ind, num_classes=log_probs.shape[1]).type_as(log_probs)
    # generate z_tilde
    z_tilde = c_phi(log_probs + u_Gumbel)
    # sample s_tilde from p(s_tilde|z), see appendix D of Tucker et al. 2017
    v_b = v[torch.arange(probs.shape[0]), z_ind].unsqueeze(1)
    v_Gumbel = -torch.log(-torch.log(v))
    v_prime = -torch.log(-(torch.log(v) / probs) - torch.log(v_b))
    v_prime[torch.arange(probs.shape[0]), z_ind] = v_Gumbel[torch.arange(probs.shape[0]), z_ind]
    s_tilde = c_phi(v_prime)
    # compute RELAX estimator (evaluate loss at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)
    f_z_tilde = loss_func(z_tilde, target)
    f_s_tilde = loss_func(s_tilde, target)
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind).unsqueeze(1)
    # detach c_phi
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
    g_estimator = (f_z - f_s_tilde) * g_log_prob + (g_f_z_tilde - g_f_s_tilde)
    var_estimator = (g_estimator ** 2).mean(1).sum()
    # backward through var estimator to optimize eta and temp
    var_estimator.backward(create_graph=True)
    return estimator.mean(1).sum()
