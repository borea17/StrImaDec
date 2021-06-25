import torch
import numpy as np
import torch.distributions as dists
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_accuracy(training_step_outputs, discrete_vae, mode="stochastic"):
    num_clusters = discrete_vae.latent_dim
    num_classes = training_step_outputs[-1]["y"].shape[1]
    # assignment problem -> build matrix
    assignment_matrix = np.zeros([num_clusters, num_classes])

    all_labels = []
    all_preds = []
    for entry in training_step_outputs:
        x, y = entry["x"], entry["y"]
        # all one hot vectors repeated current batch_size times
        batch_size = x.shape[0]
        # all pos. classes (one-hot) [batch_size, num_classes, num_classes]
        all_classes = np.eye(num_classes).reshape(1, num_classes, num_classes).repeat(batch_size, 0)
        # reshape y into [batch_size, 1, num_classes]
        y = y.unsqueeze(1).detach().cpu().numpy()
        # find target indices by finding vectors that agree with all_classes
        target_indices = np.all(y == all_classes, axis=2).argmax(1)
        # predict cluster either stochastic (sampling) or deterministic
        probs_logits = discrete_vae.encode(x)
        if mode == "stochastic":  # predictions are sampled one-hot vectors
            # sample class indices [batch]
            sampled_indices = dists.Categorical(logits=probs_logits).sample()
            # cast to one-hot vectors [batch, num_clusters]
            pred = F.one_hot(sampled_indices, num_classes=num_clusters).type_as(probs_logits)
        elif mode == "deterministic":  # predictions are the argmax of each probability dist
            pred = torch.zeros_like(probs_logits)
            pred[torch.arange(pred.shape[0]), torch.argmax(probs_logits, dim=1)] = 1
        # predicted clusters [batch_size, 1, num_clusters]
        pred = pred.unsqueeze(1).cpu().detach().numpy()
        # all possible clusters
        all_clusters = (
            np.eye(num_clusters).reshape(1, num_clusters, num_clusters).repeat(batch_size, 0)
        )
        # find source indices by finding vectors that agree with all_clusters
        source_indices = np.all(pred == all_clusters, axis=2).argmax(1)
        # inplace update of assignment_matrix
        np.add.at(assignment_matrix, (source_indices, target_indices), 1)
        # collect labels and predictions
        all_labels.append(y)
        all_preds.append(pred)
    if num_classes == num_clusters:  # one-to-one
        # find optimal assignment using hungarian method
        row_ind, col_ind = linear_sum_assignment(-assignment_matrix)
        empty = 0
    elif num_classes < num_clusters:  # many-to-one
        # number of clusters that are not assigned
        empty = (np.sum(assignment_matrix, axis=1) == 0).sum()
        # greedy approach
        col_ind = np.argmax(assignment_matrix, axis=1)
        if not np.all(np.in1d(list(range(num_classes)), col_ind)) and mode == "deterministic":
            print(f"Assertion: Greedy solution does not work for {num_clusters}")
        # assert np.all(np.in1d(list(range(num_classes)), col_ind)), "greedy solution does not work"
        row_ind = np.arange(num_clusters)
    # set matching values zero (in place)
    assignment_matrix[row_ind, col_ind] = 0
    # the rest of assignment matrix would be "misclassified"
    accuracy = 1 - np.sum(assignment_matrix) / sum([labels.shape[0] for labels in all_labels])
    return accuracy, empty