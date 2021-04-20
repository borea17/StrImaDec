import torch.distributions as dists
import torch.nn.functional as F


def NVIL(probs_logits, loss_func, baseline):
    """
        computes the loss using the NVIL estimator (REINFORCE + neural baseline)
        through which we can backpropagate, at the same time the neural baseline loss
        is computed

    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        loss_func (method): loss function that takes the sampled class vectors as input
        baseline (tensor): predicted baseline values from neural net [batch]
    """
    # get categorial distribution
    categorical_dist = dists.Categorical(logits=probs_logits)
    # sample class indices [batch]
    sampled_indices = categorical_dist.sample()
    # cast to one-hot vectors [batch, L]
    num_classes = probs_logits.shape[1]
    sampled_class = F.one_hot(sampled_indices, num_classes=num_classes).type_as(probs_logits)
    # compute loss [batch, L]
    loss = loss_func(sampled_class) - baseline.detach()
    # compute estimator [batch, L]
    estimator = loss.detach() * categorical_dist.log_prob(sampled_indices).unsqueeze(1)
    baseline_loss = (loss_func(sampled_class).detach() - baseline).pow(2)
    return estimator.sum() + baseline_loss.sum()