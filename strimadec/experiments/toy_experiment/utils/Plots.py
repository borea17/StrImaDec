from cycler import cycler

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


# use diverging colors from colorbrewer2.org
colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "gray"]
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e', "gray"]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
linestyles = 5 * ["solid"] + ["dashdot"]
fontsize, labelsize = 13, 11


class Plots:

    # from colorbrewer2.org
    COLORS = ['#377eb8','#984ea3','#ff7f00','#4daf4a', '#e41a1c', "gray"]
    LINESTYLES = 5 * ["solid"] + ["dashdot"]
    FONTSIZE, LABELSIZE = 13, 11

    @staticmethod
    def plot_toy(results, store_path_fig):
        fontsize, labelsize = Plots.FONTSIZE, Plots.LABELSIZE
        linestyles = Plots.LINESTYLES
        plt.rcParams["axes.prop_cycle"] = cycler(color=Plots.COLORS)

        if len(results) == 4:
            # use same color scheme as Grathwohl et al. (2017) for replication exp.
            colors = ["#377eb8", '#4daf4a', '#e41a1c', "gray"]
            plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
            linestyles = 3 * ["solid"] + ["dashdot"]

        fig = plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        for i, estimator_name in enumerate(results.keys()):
            losses = results[estimator_name]["losses"]  # [num_iterations, num_epochs]
            steps = np.arange(losses.shape[1]) + 1  # start from 1
            mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
            plt.plot(steps, mean_loss, linestyle=linestyles[i], label=estimator_name)
            plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        plt.xlim([min(steps), max(steps)])
        plt.ylabel("Loss", fontsize=fontsize)
        plt.xlabel("Steps", fontsize=fontsize)
        # remove top and right border, adjust labelsize of ticks
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        # add legend
        plt.legend(loc=(0.5, 0.15), fontsize=fontsize)
        # plt.legend(fontsize=fontsize)

        plt.subplot(1, 2, 2)
        for i, estimator_name in enumerate(results.keys()):
            if estimator_name == "Exact gradient":
                continue
            log_vars = np.log(results[estimator_name]["vars_grad"] + 1e-64)
            steps = np.arange(log_vars.shape[1]) + 1  # start from 1
            mu_log_vars, std_log_vars = np.mean(log_vars, axis=0), np.std(log_vars, axis=0)
            plt.plot(steps, mu_log_vars, linestyle=linestyles[i], label=estimator_name)
            plt.fill_between(
                steps, mu_log_vars - std_log_vars, mu_log_vars + std_log_vars, alpha=0.2
            )
        plt.ylabel("Log Variance of Gradient Estimates", fontsize=fontsize)
        plt.xlabel("Steps", fontsize=fontsize)
        # remove top and right border, adjust labelsize of ticks
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        # add legend
        plt.xlim([min(steps), max(steps)])
        if len(results) == 4:
            plt.ylim([-20, 0])
            plt.legend(fontsize=fontsize)
        else:
            plt.ylim([-20, 10])
            plt.legend(loc=(0.3, 0.6), fontsize=fontsize)
        # plt.legend(fontsize=fontsize)
        plt.show()
        # store figure
        fig.savefig(store_path_fig, bbox_inches="tight")
        return

    @staticmethod
    def plot_hyperparamter(results, hyperparam_name, hyperparams, store_path_fig):
        fontsize, labelsize = Plots.FONTSIZE, Plots.LABELSIZE
        plt.rcParams["axes.prop_cycle"] = cycler(color=Plots.COLORS)
        # plot exact gradient only once
        exact_gradient_plotted = False
        fig = plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        for estimator_name, data_est in results.items():
            for i, hyperparam in enumerate(hyperparams):
                losses = data_est[f"losses_{hyperparam_name}_{i}"]
                steps = np.arange(losses.shape[1]) + 1  # start from 1
                mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
                if estimator_name == "Exact gradient" and not exact_gradient_plotted:
                    plt.plot(steps, mean_loss, linestyle="dashdot", label=estimator_name, color="gray")
                    exact_gradient_plotted = True
                    continue
                elif estimator_name != "Exact gradient":
                    plt.plot(steps, mean_loss, label=f"{estimator_name} {hyperparam_name}={hyperparam}")
                    plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        plt.xlim([min(steps), max(steps)])
        plt.ylabel("Loss", fontsize=fontsize)
        plt.xlabel("Steps", fontsize=fontsize)
        # remove top and right border, adjust labelsize of ticks
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        # add legend
        plt.legend(fontsize=fontsize)
        plt.subplot(1, 2, 2)
        for estimator_name, data_est in results.items():
            for i, hyperparam in enumerate(hyperparams):
                if estimator_name == "Exact gradient":
                    continue
                vars_grad = data_est[f"vars_grad_{hyperparam_name}_{i}"]
                steps = np.arange(vars_grad.shape[1]) + 1  # start from 1
                log_vars = np.log(vars_grad + 1e-64)

                log_vars, std_vars = np.mean(log_vars, axis=0), np.std(log_vars, axis=0)

                plt.plot(steps, log_vars, label=f"{estimator_name} {hyperparam_name}={hyperparam}")
                plt.fill_between(steps, log_vars - std_vars, log_vars + std_vars, alpha=0.2)
        plt.xlim([min(steps), max(steps)])
        plt.ylabel("Log Variance of Gradient Estimates", fontsize=fontsize)
        plt.xlabel("Steps", fontsize=fontsize)
        # remove top and right border, adjust labelsize of ticks
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        # add legend
        plt.legend(fontsize=fontsize)
        plt.show()
        # store figure
        fig.savefig(store_path_fig, bbox_inches="tight")
        return


    @staticmethod
    def plot_hyperparameter_losses(results, hyperparam_name, hyperparams, store_path_fig):
        fontsize, labelsize = Plots.FONTSIZE, Plots.LABELSIZE
        plt.rcParams["axes.prop_cycle"] = cycler(color=Plots.COLORS)
        fig = plt.figure(figsize=(14, 5))
        for i, hyperparam in enumerate(hyperparams):
            plt.subplot(1, len(hyperparams), i + 1)
            for estimator_name, data_est in results.items():
                losses = data_est[f"losses_{hyperparam_name}_{i}"]
                steps = np.arange(losses.shape[1]) + 1  # start from 1
                mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
                if estimator_name == "Exact gradient":
                    plt.plot(steps, mean_loss, linestyle="dashdot", label=estimator_name, color="gray")
                    continue
                plt.plot(steps, mean_loss, label=estimator_name)
                plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
            plt.xlim([min(steps), max(steps)])
            plt.ylabel("Loss", fontsize=fontsize)
            plt.xlabel("Steps", fontsize=fontsize)
            plt.title(f"{hyperparam_name}={hyperparam}", fontsize=fontsize)
            ax = plt.gca()
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.tick_params(axis="both", which="major", labelsize=labelsize)
            plt.legend(fontsize=fontsize)
        plt.show()
        # store figure
        fig.savefig(store_path_fig, bbox_inches="tight")
        return