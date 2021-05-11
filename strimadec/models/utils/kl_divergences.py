def gaussian_kl(q, p):
    """
        computes the KL divergence per batch between two Gaussian distribution
        parameterized by q and p

    Args:
        q (list) Gaussian distribution parametrized by
            q[0] (torch tensor): mean of q [batch_size, latent_dim]
            q[1] (torch tensor): variance of q [batch_size, latent_dim]
        p (list) Gaussian distribution parametrized by
            p[0] (torch tensor): mean of p [batch_size, latent_dim]
            p[1] (torch tensor): variance of p [batch_size, latent_dim]

    Returns:
        kl_div (torch.tensor): kl divergence [batch_size]

    see https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
    """
    mean_q, var_q = q[0], q[1]
    mean_p, var_p = p[0], p[1]

    var_ratio = var_q / var_p
    t1 = (mean_q - mean_p).pow(2) / var_p
    return -0.5 * (1 + var_ratio.log() - var_ratio - t1).sum(1)