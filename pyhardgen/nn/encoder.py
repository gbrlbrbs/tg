import torch
import torch.nn as nn
from fastai.tabular.all import *
from .helpers import match_activation
from ..config import model

class Encoder(nn.Module):
    def __init__(self, n_meta_feat, latent_size: int, params: model, ps=0.2):
        """Encoder class. Inherits from `TabularModel`.

        Args:
            latent_size (int): Latent size.
            params (`model`): Parameters for the encoder.
        """
        super(Encoder, self).__init__()
        layers = params.features
        act_fn = match_activation(params.activation)
        layers = []
        for i in range(params.n_layers):
            if i == 0:
                layers.append(LinBnDrop(n_meta_feat, params.features[i], p=ps, act=act_fn))
            else:
                layers.append(LinBnDrop(params.features[i-1], params.features[i], p=ps, act=act_fn))

        layers.append(
            LinBnDrop(params.features[-1], latent_size, p=ps, act=None, bn=False)
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        return encoded