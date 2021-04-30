class DVAE_Loss:
    def __init__(self, decoder_distribution: str):
        self.decoder_distribution = decoder_distribution
        return

    def assign_VAE(self, VAE):
        self.VAE = VAE
        return

    def loss_func(self, sampled_class, x):
        x_rec = self.VAE.decode(sampled_class)

        pass