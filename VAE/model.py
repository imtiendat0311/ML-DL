
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Module, var: nn.Module) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        print("VAE model initialized")

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: torch.Tensor):
        encoder_outputs = self.encoder(x)
        mu = self.mean(encoder_outputs)
        var = self.var(encoder_outputs)
        x = self.reparameterize(mu, torch.exp(0.5 * var))
        decoder_outputs = self.decoder(x)
        return x, mu, var, encoder_outputs, decoder_outputs
