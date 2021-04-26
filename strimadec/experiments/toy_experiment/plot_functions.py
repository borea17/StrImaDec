import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def plot_toy_results(results):
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    colors = ["#1F77B4", "#2CA02C", "#D62728", "gray"]
    # start plot
    fig = plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    for i, results_dict in enumerate(results):
        losses = results_dict["expected_losses"]
        steps = np.arange(len(losses))
        name = results_dict["name"]
<<<<<<< HEAD
        plt.plot(steps, losses, label=name, color=colors[i])
=======
        if "analytical" in name:
            plt.plot(steps, losses, label=name, color="gray")
>>>>>>> main
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, results_dict in enumerate(results):
        log_vars_grad = np.log(results_dict["vars_grad"] + 1e-12)
        steps = np.arange(len(log_vars_grad))
        name = results_dict["name"]
        if name != "analytical":
            # smoothen log_var
            smooth_log_var = uniform_filter1d(log_vars_grad, size=5, mode="reflect")
            plt.plot(steps, smooth_log_var, label=name, color=colors[i])
    plt.xlabel("Steps")
    plt.ylabel("Log (Var (Gradient Estimator) )")
    plt.legend()
    plt.show()
    return