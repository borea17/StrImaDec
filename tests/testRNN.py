from unittest import TestCase

import numpy as np
import torch

from strimadec.models.modules import RNN


class testRNN(TestCase):

    BATCH_SIZE = 5
    SEED = 7

    def setUp(self):
        np.random.seed(testRNN.SEED)
        # (deterministic) random setup of RNN
        self.random_config = testRNN.build_random_config()
        return

    def test_RNN_forward_works_as_expected(self):
        batch_size = testRNN.BATCH_SIZE
        img_channels, img_dim = self.random_config["img_channels"], self.random_config["img_dim"]
        latent_space_dim = self.random_config["latent_space_dim"]
        hidden_state_dim = self.random_config["hidden_state_dim"]
        output_size = self.random_config["output_size"]

        rnn = RNN(self.random_config)
        # generate fake data
        fake_img = 10 * torch.randn([batch_size, img_channels, img_dim, img_dim])
        fake_z_im1 = torch.randn([batch_size, latent_space_dim])
        fake_h_im1 = torch.randn([batch_size, hidden_state_dim])
        # run forward
        omega_i, h_i = rnn(fake_img, fake_z_im1, fake_h_im1)
        # check shapes
        self.assertTrue(omega_i.shape == torch.Size([batch_size, output_size]))
        self.assertTrue(h_i.shape == torch.Size([batch_size, hidden_state_dim]))
        # check initialization worked
        expected_omega_i = torch.tensor(self.random_config["output_bias_init"]).repeat(
            batch_size, 1
        )
        expected_omega_i[:, 0] = torch.sigmoid(expected_omega_i[:, 0])
        self.assertTrue((expected_omega_i - omega_i).pow(2).sum() < 1e-12)
        # check that we can rerun using h_i
        fake_h_im1 = h_i
        # run forward
        omega_i, h_i = rnn(fake_img, fake_z_im1, fake_h_im1)
        # check shapes
        self.assertTrue(omega_i.shape == torch.Size([batch_size, output_size]))
        self.assertTrue(h_i.shape == torch.Size([batch_size, hidden_state_dim]))
        # check initialization worked
        expected_omega_i = torch.tensor(self.random_config["output_bias_init"]).repeat(
            batch_size, 1
        )
        expected_omega_i[:, 0] = torch.sigmoid(expected_omega_i[:, 0])
        self.assertTrue((expected_omega_i - omega_i).pow(2).sum() < 1e-12)
        return

    @staticmethod
    def build_random_config():
        z_pres_dim, z_where_dim, z_what_dim = 1, 3, np.random.randint(10, 20)
        p_pres_init = np.random.rand(1, 1)
        mean_z_where_init = np.random.randint(1, 5, size=(1, 3))
        log_var_z_where_init = np.random.randint(1, 5, size=(1, 3))
        output_bias_init = np.concatenate((p_pres_init, mean_z_where_init, log_var_z_where_init), 1)
        config = {
            "baseline_net": False,
            "img_channels": np.random.randint(1, 5),
            "img_dim": 8 * np.random.randint(4, 8),
            "hidden_state_dim": np.random.randint(25, 100),
            "latent_space_dim": z_pres_dim + z_where_dim + z_what_dim,
            "FC_hidden_dims": np.random.randint(1, 100, size=(np.random.randint(1, 4))),
            "output_size": z_pres_dim + 2 * z_where_dim,
            "output_bias_init": output_bias_init,
        }
        return config