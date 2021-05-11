from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F

from strimadec.models.modules import LocalizationNet


class testLocalizationNet(TestCase):

    BATCH_SIZE = 10
    SEED = 2

    def setUp(self):
        np.random.seed(testLocalizationNet.SEED)
        # (deterministic) random setup of localization net
        self.random_config = testLocalizationNet.build_random_config()
        return

    def test_localization_forward_works_as_expected(self):
        batch_size = testLocalizationNet.BATCH_SIZE
        config = self.random_config
        img_channels, img_dim = config["img_channels"], config["img_dim"]
        localization_net = LocalizationNet(config)

        fake_img = 20 * torch.randn(batch_size, img_channels, img_dim, img_dim)
        results = localization_net(fake_img)

        mu_t_expected = torch.tensor(config["prior_mu_transform"]).repeat(batch_size, 1)
        var_t_expected = torch.tensor(config["prior_var_transform"]).repeat(batch_size, 1)
        # check initialization works
        self.assertTrue(((results["mu_t"] - mu_t_expected) ** 2).sum() < 1e-8)
        self.assertTrue(((results["log_var_t"].exp() - var_t_expected) ** 2).sum() < 1e-8)
        return

    def test_spatial_transform_for_analytical_works_as_expected(self):
        batch_size = testLocalizationNet.BATCH_SIZE
        config = self.random_config
        img_channels, img_dim = config["img_channels"], config["img_dim"]
        localization_net = LocalizationNet(config)
        # generate fake data
        L = np.random.randint(2, 10)
        fake_prototype = 20 * torch.randn(batch_size, L, img_channels, img_dim, img_dim)
        fake_t = 20 * torch.randn(batch_size, 6)
        # compute expected output through for loop
        x_tilde_expected = torch.zeros([batch_size, L, img_channels, img_dim, img_dim])
        transformation_matrix = fake_t.view(-1, 2, 3)
        for i in range(L):
            x_p_i = fake_prototype[:, i]
            grid_i = F.affine_grid(transformation_matrix, x_p_i.size(), align_corners=False)
            x_tilde_expected[:, i] = F.grid_sample(x_p_i, grid_i, align_corners=False)
        # compute x_tilde via spatial transform
        x_tilde = localization_net.spatial_transform(fake_prototype, fake_t)
        # assert equality
        self.assertTrue(((x_tilde - x_tilde_expected) ** 2).sum() < 1e-9)
        return

    @staticmethod
    def build_random_config():
        img_channels, img_dim = np.random.randint(1, 20, size=2)
        config = {
            "img_channels": img_channels,
            "img_dim": img_dim,
            "input_dim": img_channels * img_dim * img_dim,
            "FC_hidden_dims": np.random.randint(1, 100, size=(np.random.randint(1, 4))),
            "prior_mu_transform": np.random.randint(1, 100, size=(6)),
            "prior_var_transform": np.random.randint(1, 100, size=(6)),
        }
        return config