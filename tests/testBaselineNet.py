from unittest import TestCase

import numpy as np
import torch

from strimadec.models.modules import BaselineNet


class testBaselineNet(TestCase):

    BATCH_SIZE = 10
    SEED = 1

    def setUp(self):
        np.random.seed(testBaselineNet.SEED)
        # (deterministic) random setup of baseline net
        self.random_config = testBaselineNet.build_random_config()
        return

    def test_baseline_net_works_as_NVIL_baseline(self):
        batch_size = testBaselineNet.BATCH_SIZE
        img_channels, img_dim = self.random_config["img_channels"], self.random_config["img_dim"]
        fake_img = 10 * torch.randn([batch_size, img_channels, img_dim, img_dim])

        config = self.random_config
        config["input_dim"] = img_channels * img_dim * img_dim
        config["output_dim"] = 1

        NVIL_baseline = BaselineNet(config)

        out = NVIL_baseline(fake_img)

        self.assertTrue(out.shape == torch.Size([batch_size, config["output_dim"]]))
        return

    def test_baseline_net_works_as_RELAX_baseline(self):
        batch_size = testBaselineNet.BATCH_SIZE
        num_clusters = np.random.randint(3, 10)
        fake_input = 10 * torch.randn([batch_size, num_clusters])

        config = self.random_config
        config["input_dim"] = num_clusters
        config["output_dim"] = num_clusters
        config["log_temp_init"] = 0.0

        RELAX_baseline = BaselineNet(config)

        out = RELAX_baseline(fake_input)

        self.assertTrue(out.shape == torch.Size([batch_size, config["output_dim"]]))
        self.assertTrue(RELAX_baseline.contains_temp == True)
        return

    @staticmethod
    def build_random_config():
        config = {
            "img_channels": np.random.randint(1, 5),
            "img_dim": 8 * np.random.randint(4, 16),
            "FC_hidden_dims": np.random.randint(1, 100, size=(np.random.randint(1, 4))),
        }
        return config