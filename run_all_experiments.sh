#!/bin/bash

# DVAE experiments SimplifiedMNIST
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'REINFORCE' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'NVIL' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'CONCRETE' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'REBAR' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'RELAX' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3
python strimadec/experiments/single_object_multi_class/01_DVAE_experiments.py \
    --num_epochs 1 --estimator_name 'Exact gradient' --decoder_dist 'Gaussian' \
    --dataset_name 'SimplifiedMNIST' --num_clusters 3