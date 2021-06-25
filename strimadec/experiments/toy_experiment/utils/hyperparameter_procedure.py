import numpy as np

from strimadec.experiments.toy_experiment.utils.optimization_procedure import (
    run_stochastic_optimization,
)
from strimadec.experiments.toy_experiment.utils.experimental_setup import build_experimental_setup


def run_hyperparameter_procedure(
    estimator_names, hyperparams, hyperparam_name, num_epochs, num_repetitions, target, store_dir
):
    results = {}
    for i, estimator_name in enumerate(estimator_names):
        results[estimator_name] = {}
        losses_per_hyperparam = np.zeros([num_repetitions, num_epochs])
        vars_per_hyperparam = np.zeros([num_repetitions, num_epochs])
        elapsed_times_per_hyperparam = np.zeros([num_repetitions, num_epochs])
        for i, hyperparam in enumerate(hyperparams):
            losses_per_hyperparam_i = losses_per_hyperparam.copy()
            vars_per_hyperparam_i = vars_per_hyperparam.copy()
            elapsed_times_per_hyperparam_i = elapsed_times_per_hyperparam.copy()
            for i_exp in range(num_repetitions):
                print(f"Start {estimator_name}-estimator {i_exp + 1}/{num_repetitions} ...")
                SEED = i_exp
                params = build_experimental_setup(estimator_name, target, num_epochs, SEED)
                params[hyperparam_name] = hyperparam
                current_results_dict = run_stochastic_optimization(params)
                losses_per_hyperparam_i[i_exp] = current_results_dict["expected_losses"]
                vars_per_hyperparam_i[i_exp] = current_results_dict["vars_grad"]
                elapsed_times_per_hyperparam_i[i_exp] = current_results_dict["elapsed_times"]
                if estimator_name == "Exact gradient":  # deterministic, do not run for all reps
                    losses_per_hyperparam_i = losses_per_hyperparam_i[i_exp][None, :]
                    vars_per_hyperparam_i = vars_per_hyperparam_i[i_exp][None, :]
                    elapsed_times_per_hyperparam_i = elapsed_times_per_hyperparam_i[i_exp][None, :]
                    break
            results[estimator_name].update(
                {
                    f"losses_{hyperparam_name}_{i}": losses_per_hyperparam_i,
                    f"vars_grad_{hyperparam_name}_{i}": vars_per_hyperparam_i,
                    f"elapsed_times_{hyperparam_name}_{i}": elapsed_times_per_hyperparam_i,
                }
            )
        # store current estimator results
        store_path = f"{store_dir}/hyperparameter_{hyperparam_name}_exp_{estimator_name}.npy"
        np.save(store_path, results[estimator_name])
    return results