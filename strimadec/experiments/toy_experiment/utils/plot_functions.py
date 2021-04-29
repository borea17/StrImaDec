from cycler import cycler

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def plot_toy(results, store_path_fig):
    # use diverging colors from colorbrewer2.org
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "gray"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    linestyles = 5 * ["solid"] + ["dashdot"]
    fontsize = 13
    markers = 6 * [None]
    fig = plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for i, estimator_name in enumerate(results.keys()):
        losses = results[estimator_name]["losses"]  # [num_iterations, num_epochs]
        steps = np.arange(losses.shape[1])
        mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
        plt.plot(steps, mean_loss, linestyle=linestyles[i], label=estimator_name)
        plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    plt.xlim([min(steps), max(steps) + 1])
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Steps", fontsize=fontsize)
    plt.show()
    # store figure
    fig.savefig(store_path_fig, bbox_inches="tight")
    return


def plot_replication(results, store_path_fig):
    # same color scheme as Grathwohl et al., 2018
    colors = ["#1F77B4", "#2CA02C", "#D62728", "gray"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    linestyles = 3 * ["solid"] + ["dashdot"]
    # start plot
    fig = plt.figure(figsize=(14, 5))
    fontsize, labelsize = 13, 11
    plt.subplot(1, 2, 1)
    for i, results_dict in enumerate(results):
        losses = results_dict["expected_losses"]
        steps = np.arange(len(losses))
        name = results_dict["name"]
        plt.plot(steps, losses, linestyle=linestyles[i], label=name)
    plt.xlim([min(steps), max(steps) + 1])
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Steps", fontsize=fontsize)
    # remove top and right border, adjust labelsize of ticks
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    # add legend
    legend = plt.legend(loc=(0.6, 0.1), fontsize=fontsize)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_linewidth(0.0)

    plt.subplots_adjust(wspace=0.3)

    plt.subplot(1, 2, 2)
    for i, results_dict in enumerate(results):
        log_vars_grad = np.log(results_dict["vars_grad"] + 1e-32)
        steps = np.arange(len(log_vars_grad))
        name = results_dict["name"]
        if name != "Exact gradient":
            # smoothen log_var
            smooth_log_var = uniform_filter1d(log_vars_grad, size=10, mode="reflect")
            plt.plot(steps, smooth_log_var, linestyle=linestyles[i], label=name)
    plt.xlim([min(steps), max(steps) + 1])
    plt.xlabel("Steps", fontsize=fontsize)
    plt.ylabel("Log Variance of Gradient Estimates", fontsize=fontsize)
    # remove top and right border
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    # add legend
    legend = plt.legend(loc=(0.05, 0.61), fontsize=fontsize)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_linewidth(0.0)
    plt.show()
    # store figure
    fig.savefig(store_path_fig, bbox_inches="tight")
    return