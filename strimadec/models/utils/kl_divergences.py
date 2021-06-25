import torch


def gaussian_kl(q, p):
    """
        computes the KL divergence per batch between two Gaussian distributions
        parameterized by q and p

    Args:
        q (list) Gaussian distribution parametrized by
            q[0] (torch tensor): mean of q [batch_size, *latent_dims]
            q[1] (torch tensor): variance of q [batch_size, *latent_dims]
        p (list) Gaussian distribution parametrized by
            p[0] (torch tensor): mean of p [batch_size, *latent_dims]
            p[1] (torch tensor): variance of p [batch_size, *latent_dims]

    Returns:
        kl_div (torch.tensor): kl divergence [batch_size, *latent_dims]

    see https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
    """
    mean_q, var_q = q[0], q[1]
    mean_p, var_p = p[0], p[1]

    var_ratio = var_q / var_p
    t1 = (mean_q - mean_p).pow(2) / var_p
    return -0.5 * (1 + var_ratio.log() - var_ratio - t1)


def bernoulli_kl(q_probs, p_probs):
    """
        computes the KL divergence per batch between two Bernoulli distributions
        parametrized by q_probs and p_probs

    Note: EPS is added for numerical stability, see
    https://github.com/pytorch/pytorch/issues/15288

    Args:
        q_probs (torch tensor): mean of q [batch_size, *latent_dims]
        p_probs (torch tensor): mean of p [batch_size, *latent_dims]

    Returns:
        kl_div (torch tensor): kl divergence [batch_size, *latent_dims]
    """
    EPS = 1e-32
    p1 = p_probs
    p0 = 1 - p1
    q1 = q_probs
    q0 = 1 - q1

    logq1 = (q1 + EPS).log()
    logq0 = (q0 + EPS).log()
    logp1 = (p1).log()
    logp0 = (p0).log()

    kl_div_1 = q1 * (logq1 - logp1)
    kl_div_0 = q0 * (logq0 - logp0)
    return kl_div_1 + kl_div_0


def categorical_kl(q_logits, p_logits):
    """
        computes the KL divergence per batch between two Categorical distributions
        parameterized by q and p


    Args:
        q_logits (torch tensor): logits of q dist [batch_size, *latent_dims]
        p_logits (torch tensor): logits of p dist [batch_size, *latent_dims]

    Returns:
        kl_div (torch tensor): kl divergence [batch_size, *latent_dims]

    see https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
    """
    import pdb 
    pdb.set_trace()
    q_probs = torch.softmax(q_logits, dim=-1)

    t = q_probs * (q_logits - p_logits)
    t[(q_probs == 0).expand_as(t)] = 0

    return t