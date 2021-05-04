import torch.distributions as dists
import torch.nn.functional as F


def NVIL(probs_logits, target, baseline_vals, loss_func):
    """
        computes the loss using the NVIL estimator (REINFORCE + neural baseline)
        through which we can backpropagate, at the same time the neural baseline loss
        is computed

    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        target (tensor): target tensor [batch, L]
        baseline_vals (tensor): predicted baseline values from neural net [batch, 1]
        loss_func (method): loss function that takes the sampled class vectors and target as input

    Returns:
        estimator (tensor): batch-wise loss (including baseline_loss when backward) [batch]
    """
    # get categorial distribution
    categorical_dist = dists.Categorical(logits=probs_logits)
    # sample class indices [batch]
    sampled_indices = categorical_dist.sample()
    # cast to one-hot vectors [batch, L]
    num_classes = probs_logits.shape[1]
    sampled_class = F.one_hot(sampled_indices, num_classes=num_classes).type_as(probs_logits)
    # compute loss [batch, L] and loss_with_control_variate [batch, L]
    loss = loss_func(sampled_class, target)
    loss_with_control_variate = (loss - baseline_vals).detach()
    # compute estimator [batch, L]
    estimator = loss_with_control_variate * categorical_dist.log_prob(sampled_indices).unsqueeze(1)
    baseline_loss = (loss.detach() - baseline_vals).pow(2)
    return estimator.sum(1) + baseline_loss.sum(1) - baseline_loss.detach().sum(1), loss.sum(1)