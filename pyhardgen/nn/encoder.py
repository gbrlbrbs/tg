import torch
import torch.nn as nn
from fastai.tabular.all import *
from .helpers import match_activation
from ..config import model

class Encoder(TabularModel):
    def __init__(self, emb_szs, n_cont, latent_size: int, params: model, low: torch.Tensor, high: torch.Tensor, ps=0.2, embed_p=0.01):
        """Encoder class. Inherits from `TabularModel`.

        Args:
            latent_size (int): Latent size.
            params (`model`): Parameters for the encoder.
        """
        layers = params.features
        act_fn = match_activation(params.activation)
        super().__init__(emb_szs, n_cont, latent_size, layers, embed_p=embed_p, act_cls=act_fn)
        self.layers = nn.Sequential(
            *L(*self.layers.children())[:-1] + 
            nn.Sequential(LinBnDrop(layers[-1], latent_size, act=None, bn=False, p=ps), SigmoidRange(low=low, high=high))
            )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = super().forward(x_cat, x_cont)
        return encoded