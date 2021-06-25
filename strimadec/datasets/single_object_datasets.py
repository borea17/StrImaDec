import os

import torch
import git
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder

from strimadec.datasets.utils import get_git_root


class FashionMNIST(datasets.FashionMNIST):

    """standard FashionMNIST dataset with one-hot vectors as labels instead of class labels"""

    def __init__(self, train):
        git_root_path = get_git_root(os.getcwd())
        store_dir = os.path.join(git_root_path, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        super().__init__(
            root=store_dir,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        return
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 10 classes)
        target = torch.eye(10)[target]
        return img, target


class FullMNIST(datasets.MNIST):
    """standard MNIST dataset with one-hot vectors as labels instead of class labels"""

    def __init__(self, train):
        git_root_path = get_git_root(os.getcwd())
        store_dir = os.path.join(git_root_path, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        super(FullMNIST, self).__init__(store_dir, train, transform, target_transform, download)
        return

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 10 classes)
        target = torch.eye(10)[target]
        return img, target


class Letters(datasets.EMNIST):
    """26 letters dataset with one-hot vectors as labels"""

    def __init__(self) -> None:
        git_root_path = get_git_root(os.getcwd())
        store_dir = os.path.join(git_root_path, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        split = "letters"
        super().__init__(
            root=store_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        return

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 26 classes)
        target = torch.eye(26)[target - 1]  # target goes from 1 to 26
        return img, target


class SimplifiedMNIST(torch.utils.data.TensorDataset):
    """contains only a subset of MNIST digits with one-hot vectors as labels"""

    def __init__(self, train, digits):
        git_root_path = get_git_root(os.getcwd())
        store_dir = os.path.join(git_root_path, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        MNIST_dataset = datasets.MNIST(store_dir, train, transform, target_transform, download)
        # select only specific digits
        data = []
        labels = []
        for digit in digits:
            indices_digits = MNIST_dataset.targets == digit
            torch_imgs = [
                transforms.ToTensor()(Image.fromarray(img.numpy(), mode="L"))
                for img in MNIST_dataset.data[indices_digits]
            ]
            data.append(torch.vstack(torch_imgs))
            labels.extend([digit] * sum(indices_digits))
        # vertical stack torch tensors within data list
        data = torch.vstack(data).unsqueeze(1)
        # create one-hot encoded labels
        labels = OneHotEncoder().fit_transform(np.array(labels).reshape(-1, 1)).toarray()
        # define tensor dataset
        super(SimplifiedMNIST, self).__init__(data, torch.from_numpy(labels))
        return