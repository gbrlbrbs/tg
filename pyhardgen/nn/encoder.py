import torch
import torch.nn as nn
from ..config import model

class Encoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int, params: model):
        """Encoder class.

        Args:
            input_size (int): Input size.
            latent_size (int): Latent size.
            params (`nn`): Parameters for the encoder.
        """
        super(Encoder, self).__init__()
        num_layers = params.num_layers
        features = params.features
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, features[i]))
            else:
                layers.append(nn.Linear(features[i-1], features[i]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(features[-1], latent_size))
        layers.append(nn.ReLU(True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        return encoded