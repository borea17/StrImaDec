import torch
import numpy as np

from strimadec.experiments.toy_experiment import run_stochastic_optimization, plot_toy_results


def run_experiment(run=True):
    """
        executes the toy experiment of the thesis, i.e., a 3-class categorical distribution is
        used to compare the different discrete gradient estimators

    Args:
        run (bool): decides whether experiment is executed or stored results are used
    """
    estimator_names = ["REINFORCE", "NVIL", "CONCRETE", "REBAR", "RELAX", "analytical"]
    pass