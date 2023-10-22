import torch.nn as nn
import torch
from .encoder import Encoder
from .decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """Autoencoder class.

        Args:
            encoder (Encoder): Encoder object.
            decoder (Decoder): Decoder object.
        """
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
