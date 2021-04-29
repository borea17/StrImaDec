import torch.distributions as dists
import torch.nn.functional as F


def REINFORCE(probs_logits, target, loss_func):
    """
        computes the loss such that a REINFORCE gradient estimator
        can be computed on the backward pass

    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        target (tensor): target tensor [batch, L]
        loss_func (method): loss function that takes the sampled class vectors and target as input

    Returns:
        estimator (tensor): batch-wise loss [batch]
    """
    # get categorial distribution
    categorical_dist = dists.Categorical(logits=probs_logits)
    # sample class indices [batch]
    sampled_indices = categorical_dist.sample()
    # cast to one-hot vectors [batch, L]
    num_classes = probs_logits.shape[1]
    sampled_class = F.one_hot(sampled_indices, num_classes=num_classes).type_as(probs_logits)
    # compute loss [batch, L]
    loss = loss_func(sampled_class, target)
    # compute estimator [batch, L]
    estimator = loss.detach() * categorical_dist.log_prob(sampled_indices).unsqueeze(1)
    return estimator.sum(1)
