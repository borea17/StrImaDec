import os
import pathlib

import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder


class FullMNIST(datasets.MNIST):
    """standard MNIST datasets with one-hot vectors as labels instead of class labels"""

    def __init__(self, train):
        path_to_file = "/".join(os.path.dirname(__file__).split("/")[:-7])
        store_dir = os.path.join(path_to_file, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        super(FullMNIST, self).__init__(store_dir, train, transform, target_transform, download)
        return

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 10 classes)
        target = torch.eye(10)[target]
        return img, target


class LetterDataset(datasets.EMNIST):
    """26 letters dataset with one-hot vectors as labels"""

    def __init__(self) -> None:
        path_to_file = "/".join(os.path.dirname(__file__).split("/")[:-7])
        store_dir = os.path.join(path_to_file, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        split = "letters"
        target_transform, download = None, True
        super().__init__(store_dir, split, transform, target_transform, download)
        return

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # convert target to one hot vector (there are 26 classes)
        target = torch.eye(26)[target]
        return img, target


class SimplifiedMNIST(torch.utils.data.TensorDataset):
    """contains only a subset of MNIST digits with one-hot vectors as labels"""

    def __init__(self, train, digits):
        path_to_file = "/".join(os.path.dirname(__file__).split("/")[:-7])
        store_dir = os.path.join(path_to_file, "strimadec/datasets/data")
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


if __name__ == "__main__":
    simplifiedMnist = SimplifiedMNIST(True, [2, 6, 9])