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
        probs (tensor): categorical probabilities [batch, L]
        target (tensor): target tensor [batch, L]
        temp (tensor): temperature (tuneable) to use for the concrete distribution [1]
        eta (tensor): hyperparamer that scales the control variate [1]
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
    z = F.one_hot(z_ind, num_classes=log_probs.shape[1]).type_as(log_probs)
    # Gumbel Softmax to obtain relaxed latent p(z_tilde|x)
    z_tilde = torch.softmax((log_probs + u_Gumbel) / temp, dim=1)
    # sample s_tilde from p(s_tilde|z), see appendix D of Tucker et al. 2017
    v_b = v[torch.arange(probs.shape[0]), z_ind].unsqueeze(1)
    v_Gumbel = -torch.log(-torch.log(v))
    v_prime = -torch.log(-(torch.log(v) / probs) - torch.log(v_b))
    v_prime[torch.arange(probs.shape[0]), z_ind] = v_Gumbel[torch.arange(probs.shape[0]), z_ind]
    s_tilde = torch.softmax(v_prime / temp, dim=1)
    # compute REBAR estimator (evaluate loss_func at discrete, relaxed & conditioned relaxed input)
    f_z = loss_func(z, target)
    f_z_tilde = loss_func(z_tilde, target)
    f_s_tilde = loss_func(s_tilde, target)
    log_prob = dists.Categorical(probs=probs).log_prob(z_ind).unsqueeze(1)
    # compute gradient estimator (detach eta and temp such that backward won't affect those)
    estimator = (f_z - eta * f_s_tilde).detach() * log_prob + eta.detach() * (
        f_z_tilde - f_s_tilde
    ).detach()
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
    g_estimator = (f_z - eta * f_s_tilde) * g_log_prob + eta * (g_f_z_tilde - g_f_s_tilde)
    # compute variance estimator [batch, L]
    var_estimator = g_estimator ** 2
    # # backward through var estimator to optimize eta and temp
    # var_estimator.backward(create_graph=True)
    return estimator.sum(1) + var_estimator.sum(1)