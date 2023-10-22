import torch
import torch.nn as nn
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
        num_layers = params.num_layers
        features = params.features.reverse()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_size, features[i]))
            else:
                layers.append(nn.Linear(features[i-1], features[i]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(features[-1], output_size))
        layers.append(nn.ReLU(True))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(x)
        return decoded