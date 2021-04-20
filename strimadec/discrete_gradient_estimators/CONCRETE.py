import torch.distributions as dists


def CONCRETE(probs_logits, loss_func, temp):
    """
        computes a surrogate loss through which we can backpropgate using samples 
        from a relaxed categorical (concrete) distribution
        
    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        loss_func (method): loss function that takes the sampled class vectors as input
        temp (float): temperature to use for the concrete distribution
    """
    # get concrete distribution
    concrete_dist = dists.RelaxedOneHotCategorical(temperature=temp, logits=probs_logits)
    # sample from concrete distribution with reparam trick [batch, L]
    continuous_samples = concrete_dist.rsample()
    # compute loss [batch, L]
    loss = loss_func(continuous_samples)
    # compute estimator [batch, L]
    estimator = loss
    return estimator.mean()
