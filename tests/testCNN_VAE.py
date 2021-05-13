from unittest import TestCase
import numpy as np
import torch
from strimadec.models.modules import CNN_VAE


class testCNN_VAE(TestCase):

    BATCH_SIZE = 5
    SEED = 1

    def setUp(self):
        np.random.seed(testCNN_VAE.SEED)
        # (deterministic) random network setup
        self.config = testCNN_VAE.build_random_network_config()
        return

    def test_discrete_CNN_VAE_works_as_expected(self):
        config = self.config
        config["encoder_distribution"] = "Categorical"
        decoder_distributions = ["Bernoulli", "Gaussian"]
        for decoder_dist in decoder_distributions:
            # define discrete VAE
            config["decoder_distribution"] = decoder_dist
            discrete_vae = CNN_VAE(config)
            # define input x
            img_channels, img_dim = config["img_channels"], config["img_dim"]
            x = torch.rand(testCNN_VAE.BATCH_SIZE, img_channels, img_dim, img_dim)
            # feed x through network
            results = discrete_vae(x)
            # check that outputs are as expected
            self.assertTrue("x_tilde" in results)
            self.assertTrue(
                results["x_tilde"].shape
                == torch.Size([testCNN_VAE.BATCH_SIZE, img_channels, img_dim, img_dim])
            )
            if decoder_dist == "Bernoulli":
                self.assertTrue(results["x_tilde"].min() >= 0 and results["x_tilde"].max() <= 1)
            self.assertTrue("probs_logits" in results)
            self.assertTrue(
                results["probs_logits"].shape
                == torch.Size([testCNN_VAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("z" in results)
            self.assertTrue(
                results["z"].shape == torch.Size([testCNN_VAE.BATCH_SIZE, config["latent_dim"]])
            )
        return

    def test_continuous_CNN_VAE_works_as_expected(self):
        config = self.config
        config["encoder_distribution"] = "Gaussian"
        config["fixed_var"] = np.random.rand()
        decoder_distributions = ["Bernoulli", "Gaussian"]
        for decoder_dist in decoder_distributions:
            # define discrete VAE
            config["decoder_distribution"] = decoder_dist
            discrete_vae = CNN_VAE(config)
            # define input x
            img_channels, img_dim = config["img_channels"], config["img_dim"]
            x = torch.rand(testCNN_VAE.BATCH_SIZE, img_channels, img_dim, img_dim)
            # feed x through network
            results = discrete_vae(x)
            # check that outputs are as expected
            self.assertTrue("x_tilde" in results)
            self.assertTrue(
                results["x_tilde"].shape
                == torch.Size([testCNN_VAE.BATCH_SIZE, img_channels, img_dim, img_dim])
            )
            if decoder_dist == "Bernoulli":
                self.assertTrue(results["x_tilde"].min() >= 0 and results["x_tilde"].max() <= 1)
            self.assertTrue("mu_E" in results)
            self.assertTrue(
                results["mu_E"].shape == torch.Size([testCNN_VAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("log_var_E" in results)
            self.assertTrue(
                results["log_var_E"].shape
                == torch.Size([testCNN_VAE.BATCH_SIZE, config["latent_dim"]])
            )
            self.assertTrue("z" in results)
            self.assertTrue(
                results["z"].shape == torch.Size([testCNN_VAE.BATCH_SIZE, config["latent_dim"]])
            )
        return

    @staticmethod
    def build_random_network_config():
        config = {
            "img_channels": np.random.randint(1, 5),
            "img_dim": 8 * np.random.randint(4, 8),
            "num_conv_layers_enc": np.random.randint(1, 5),
            "base_channel_enc": np.random.randint(10, 30),
            "max_channel_multiplier_enc": np.random.randint(1, 5),
            "FC_hidden_dims_enc": np.random.randint(10, 50, size=(np.random.randint(1, 5))),
            "latent_dim": np.random.randint(2, 10),
            "num_conv_layers_dec": np.random.randint(1, 5),
            "base_channel_dec": np.random.randint(10, 30),
            "max_channel_multiplier_dec": np.random.randint(1, 5),
            "fixed_var": 1,
        }
        return config