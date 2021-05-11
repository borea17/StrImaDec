import os

import torch
import numpy as np
from torchvision import transforms, datasets


class MultiMNIST(torch.utils.data.TensorDataset):

    """simple multi MNIST dataset with 0, 1, 2 non-overlapping digits"""

    def __init__(self):
        # FIXED VARIABLES
        N_SAMPLES, CANVAS_SIZE, MNIST_SIZE, SEED = 30000, 64, 28, 1
        # make dataset construction deterministic
        np.random.seed(SEED)
        train = True
        path_to_file = "/".join(os.path.dirname(__file__).split("/")[:-7])
        store_dir = os.path.join(path_to_file, "strimadec/datasets/data")
        transform = transforms.ToTensor()
        target_transform, download = None, True
        MNIST = datasets.MNIST(store_dir, train, transform, target_transform, download)
        if os.path.exists(os.path.join(store_dir), "MultiMNIST.pth"):
            tensors = torch.load(os.path.join(store_dir), "MultiMNIST.pth")
            data, labels = tensors["data"], tensors["labels"]
        else:
            data, labels = MultiMNIST.generate_dataset(N_SAMPLES, MNIST, CANVAS_SIZE, MNIST_SIZE)
        # define tensor dataset
        super(MultiMNIST, self).__init__(data, labels)
        return

    @staticmethod
    def generate_dataset(n_samples, original_MNIST, CANVAS_SIZE, MNIST_SIZE):

        data = torch.zeros([n_samples, 1, CANVAS_SIZE, CANVAS_SIZE])
        labels = torch.zeros([n_samples, 1, CANVAS_SIZE, CANVAS_SIZE])

        pos_positions = np.arange(int(MNIST_SIZE / 2), CANVAS_SIZE - int(MNIST_SIZE / 2))

        rnd_mnist_indices = np.random.randint(len(original_MNIST), size=(n_samples, 2))
        num_digits = np.random.randint(3, size=(n_samples))
        positions_0 = np.random.choice(pos_positions, size=(n_samples, 2), replace=True)
        for i_data in range(n_samples):
            if num_digits[i_data] > 0:
                # add random digit at random position
                random_digit = original_MNIST[rnd_mnist_indices[i_data][0]][0]
                x_0, y_0 = positions_0[i_data][0], positions_0[i_data][0]
                x = [x_0 - int(MNIST_SIZE / 2), x_0 + int(MNIST_SIZE / 2)]
                y = [y_0 - int(MNIST_SIZE / 2), y_0 + int(MNIST_SIZE / 2)]
                data[i_data, :, y[0] : y[1], x[0] : x[1]] += random_digit
                labels[i_data, :, y[0] : y[1], x[0] : x[1]] = 1
                if num_digits[i_data] > 1:
                    # add second non-overlapping random digit
                    random_digit = original_MNIST[rnd_mnist_indices[i_data][1]][0]
                    impos_x_pos = np.arange(x_0 - int(MNIST_SIZE / 2), x_0 + int(MNIST_SIZE / 2))
                    impos_y_pos = np.arange(y_0 - int(MNIST_SIZE / 2), y_0 + int(MNIST_SIZE / 2))
                    x_1 = np.random.choice(np.setdiff1d(pos_positions, impos_x_pos), size=1)[0]
                    y_1 = np.random.choice(np.setdiff1d(pos_positions, impos_y_pos), size=1)[0]
                    x = [x_1 - int(MNIST_SIZE / 2), x_1 + int(MNIST_SIZE / 2)]
                    y = [y_1 - int(MNIST_SIZE / 2), y_1 + int(MNIST_SIZE / 2)]
                    data[i_data, :, y[0] : y[1], x[0] : x[1]] += random_digit
                    labels[i_data, :, y[0] : y[1], x[0] : x[1]] = 2
        return data.type(torch.float32), labels