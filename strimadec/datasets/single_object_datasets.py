import os
import pathlib

import torch
from torchvision import datasets


class MNIST(datasets.MNIST):
    def __init__(self, train, transform=None, target_transform=None, download=False):
        store_dir = os.path.join(pathlib.Path(__file__).resolve().parents[0], "data")
        super(MNIST, self).__init__(store_dir, train, transform, target_transform, download)
        return

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 10 classes)
        target = torch.eye(10)[target]
        return img, target


if __name__ == "__main__":
    train_ds = MNIST(train=True, download=True)
    train_ds[0]