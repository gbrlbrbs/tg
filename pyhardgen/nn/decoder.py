import torch
import torch.nn as nn
from fastai.tabular.all import *
from .helpers import match_activation
from ..config import model

class Decoder(nn.Module):
    def __init__(self, latent_size: int, n_meta_feat: int, params: model):
        """Decoder class.
        
        Args:
            latent_size (int): Latent size.
            output_size (int): Output size.
            params (`nn`): Parameters for the decoder.
        """
        super(Decoder, self).__init__()
        num_layers = params.n_layers
        features = params.features.copy()
        features.reverse()
        activation = match_activation(params.activation)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(LinBnDrop(latent_size, features[i], p=params.dropout, act=activation))
            else:
                layers.append(LinBnDrop(features[i-1], features[i], p=params.dropout, act=activation))

        layers.append(
            LinBnDrop(features[-1], n_meta_feat, p=params.dropout, act=None, bn=False)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(x)
        return decoded