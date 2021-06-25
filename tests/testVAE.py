from unittest import TestCase

import numpy as np
import torch

from strimadec.models.modules import VAE


class testVAE(TestCase):

    BATCH_SIZE = 5
    SEED = 42

    def setUp(self):
        np.random.seed(testVAE.SEED)
        # (deterministic) random network setup
        self.config = testVAE.build_random_network_config()
        return

    def test_discrete_VAE_works_as_expected(self):
        config = self.config
        config["encoder_distribution"] = "Categorical"
        decoder_distributions = ["Bernoulli", "Gaussian"]
        for decoder_dist in decoder_distributions:
            # define discrete VAE
            config["decoder_distribution"] = decoder_dist
            discrete_vae = VAE(config)
            # define input x
            img_channels, img_dim = config["img_channels"], config["img_dim"]
            x = torch.rand(testVAE.BATCH_SIZE, img_channels, img_dim, img_dim)
            # feed x through network
            results = discrete_vae(x)
            # check that outputs are as expected
            self.assertTrue("x_tilde" in results)
            self.assertTrue(
                results["x_tilde"].shape
                == torch.Size([testVAE.BATCH_SIZE, img_channels, img_dim, img_dim])
            )
            if decoder_dist == "Bernoulli":
                self.assertTrue(results["x_tilde"].min() >= 0 and results["x_tilde"].max() <= 1)
            self.assertTrue("probs_logits" in results)
            self.assertTrue(
                results["probs_logits"].shape
                == torch.Size([testVAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("z" in results)
            self.assertTrue(
                results["z"].shape == torch.Size([testVAE.BATCH_SIZE, config["latent_dim"]])
            )
        return

    def test_continuous_VAE_works_as_expected(self):
        config = self.config
        config["encoder_distribution"] = "Gaussian"
        config["fixed_var"] = np.random.rand()
        decoder_distributions = ["Bernoulli", "Gaussian"]
        for decoder_dist in decoder_distributions:
            # define discrete VAE
            config["decoder_distribution"] = decoder_dist
            discrete_vae = VAE(config)
            # define input x
            img_channels, img_dim = config["img_channels"], config["img_dim"]
            x = torch.rand(testVAE.BATCH_SIZE, img_channels, img_dim, img_dim)
            # feed x through network
            results = discrete_vae(x)
            # check that outputs are as expected
            self.assertTrue("x_tilde" in results)
            self.assertTrue(
                results["x_tilde"].shape
                == torch.Size([testVAE.BATCH_SIZE, img_channels, img_dim, img_dim])
            )
            if decoder_dist == "Bernoulli":
                self.assertTrue(results["x_tilde"].min() >= 0 and results["x_tilde"].max() <= 1)
            self.assertTrue("mu_E" in results)
            self.assertTrue(
                results["mu_E"].shape == torch.Size([testVAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("log_var_E" in results)
            self.assertTrue(
                results["log_var_E"].shape == torch.Size([testVAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("z" in results)
            self.assertTrue(
                results["z"].shape == torch.Size([testVAE.BATCH_SIZE, config["latent_dim"]])
            )
        return

    def test_discrete_VAE_with_three_dimensional_latent_sample_works_as_expected(self):
        # define discrete VAE
        config = self.config
        config["encoder_distribution"] = "Categorical"
        config["decoder_distribution"] = "Bernoulli"
        discrete_vae = VAE(config)
        # define z [batch_size, latent_dim, latent_dim]
        batch_size, latent_dim = testVAE.BATCH_SIZE, config["latent_dim"]
        z = torch.eye(latent_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        # make loop feeding to compute x_tilde_expected
        img_channels, img_dim = config["img_channels"], config["img_dim"]
        x_tilde_expected = torch.zeros([batch_size, latent_dim, img_channels, img_dim, img_dim])
        for i_lat in range(latent_dim):
            x_tilde_expected[:, i_lat] = discrete_vae.decode(z[:, i_lat])
        # compute x_tilde via discrete vae decode
        x_tilde = discrete_vae.decode(z)
        # assert that they are equal
        self.assertTrue(((x_tilde_expected - x_tilde) ** 2).sum() < 1e-9)
        return

    @staticmethod
    def build_random_network_config():
        config = {
            "img_channels": np.random.randint(1, 5),
            "img_dim": np.random.randint(4, 8),
            "FC_hidden_dims_enc": np.random.randint(10, 50, size=(np.random.randint(1, 5))),
            "FC_hidden_dims_dec": np.random.randint(10, 50, size=(np.random.randint(1, 5))),
            "latent_dim": np.random.randint(2, 10),
            "fixed_var": 1,
        }
        return config