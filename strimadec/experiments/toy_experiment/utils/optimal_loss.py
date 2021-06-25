import torch


def compute_optimal_loss(target, loss_func):
    num_classes = target.shape[1]
    possible_losses = torch.zeros(num_classes)
    for class_ind in range(num_classes):
        one_hot_class = torch.eye(num_classes)[class_ind].unsqueeze(0).to(target.device)
        possible_losses[class_ind] = loss_func(one_hot_class, target)
    optimal_loss = min(possible_losses)
    return optimal_loss