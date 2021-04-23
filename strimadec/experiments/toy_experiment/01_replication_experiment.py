import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage.filters import uniform_filter1d

from strimadec.experiments.toy_experiment import run_stochastic_optimization