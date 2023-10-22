import torch
import torch.nn as nn
from .helpers import match_activation
from ..config import model

class Decoder(nn.Module):
    def __init__(self, latent_size: int, output_size: int, params: model):
        """Decoder class.
        
        Args:
            latent_size (int): Latent size.
            output_size (int): Output size.
            params (`nn`): Parameters for the decoder.
        """
        super(Decoder, self).__init__()
        num_layers = params.n_layers
        features = params.features.reverse()
        activation = match_activation(params.activation)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_size, features[i]))
            else:
                layers.append(nn.Linear(features[i-1], features[i]))
            layers.append(activation)
        layers.append(nn.Linear(features[-1], output_size))
        layers.append(activation)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(x)
        return decoded