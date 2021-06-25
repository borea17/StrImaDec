import torch.distributions as dists


def CONCRETE(probs_logits, target, temp, loss_func):
    """
        computes a surrogate loss through which we can backpropgate using samples
        from a relaxed categorical (concrete) distribution

    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        target (tensor): target tensor [batch, L]
        temp (float): temperature to use for the concrete distribution
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        continuous_loss (tensor): batch-wise loss [batch]
    """
    # get concrete distribution
    concrete_dist = dists.RelaxedOneHotCategorical(temperature=temp, logits=probs_logits)
    # sample from concrete distribution with reparam trick [batch, L]
    continuous_samples = concrete_dist.rsample()
    # evaluate the loss_func at the continuous samples [batch]
    continuous_loss = loss_func(continuous_samples, target)
    return continuous_loss
