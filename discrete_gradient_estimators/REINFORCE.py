import torch.distributions as dists


def REINFORCE(probs, sampled_class_ind, loss, **kwargs):
    """
        computes the the loss such that a REINFORCE gradient estimator
        can be computed on the backward pass

    Args:
        probs (tensor): categorical probabilities (output of encoder) [batch, L]
        sampled_class_ind (tensor): associated samples from Categorical dist [batch]
        loss (tensor): loss function that is applied to
    """
    categorical_dist = dists.Categorical(probs=probs)
    estimator = loss.detach() * categorical_dist.log_prob(sampled_class_ind)
    return estimator.mean()
