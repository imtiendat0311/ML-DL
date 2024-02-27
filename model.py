
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder, mean, var) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        print("VAE model initialized")

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var).to(var.device)
        return mean + var*eps

    def forward(self, x: torch.Tensor):
        encoder_outputs = self.encoder(x)
        mu = self.mean(encoder_outputs)
        log_var = self.var(encoder_outputs)
        z = self.reparameterize(mu, torch.exp(log_var*0.5))
        decoder_outputs = self.decoder(z)
        return z, mu, log_var, encoder_outputs, decoder_outputs


class Generator(nn.Module):
    def __init__(self, layer: nn.Module):
        super(Generator, self).__init__()
        self.layer = layer
        print("Generator model initialized")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Discriminator(nn.Module):
    def __init__(self, layer: nn.Module):
        super(Discriminator, self).__init__()
        self.layer = layer
        print("Discriminator model initialized")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        print("AutoEncoder model initialized")

    def forward(self, x: torch.Tensor):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return encoder_outputs, decoder_outputs
