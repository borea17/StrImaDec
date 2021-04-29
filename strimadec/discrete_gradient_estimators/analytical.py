import torch


def analytical(probs_logits, target, loss_func):
    """
        computes the expected loss through which we can easily backpropagate

    Args:
        probs_logits (tensor): categorical probabilities in logits [batch, L]
        target (tensor): target tensor [batch, L]
        loss_func (method): loss function that takes the sampled class vectors as input

    Returns:
        expected_loss (tensor): batch-wise loss [batch]
    """
    # retrieve probabilities through softmax [batch, L]
    probs = torch.softmax(probs_logits, dim=1)
    # get batch_size one-hot vectors [batch, L, L]
    one_hot_vectors = torch.eye(probs.shape[1]).unsqueeze(0).repeat(probs.shape[0], 1, 1)
    # compute loss for each one_hot_vector [batch, L]
    loss_per_one_hot = loss_func(one_hot_vectors.to(probs.device), target.unsqueeze(1)).sum(2)
    # compute expected loss by multiplying with probs [batch]
    expected_loss = (probs * loss_per_one_hot).sum(1)
    return expected_loss
