#!/bin/bash

# Toy Experiments
python strimadec/experiments/toy_experiment/01_replication_experiment.py \
    --num_epochs 10000 --num_repetitions 50
python strimadec/experiments/toy_experiment/02_toy_experiment.py \
    --num_epochs 5000 --num_repetitions 50

# Single-Object-Multi-Class Experiments
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'SimplifiedMNIST'
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'FullMNIST'
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'Letters'
python strimadec/experiments/single_object_multi_class/02_DVAEST_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'SimplifiedMNIST'
python strimadec/experiments/single_object_multi_class/02_DVAEST_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'FullMNIST'
python strimadec/experiments/single_object_multi_class/02_DVAEST_experiments.py \
    --num_epochs 150 --num_repetitions 25 --dataset_name 'Letters'

# Multi-Object-Multi-Class Experiments