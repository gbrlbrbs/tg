import torch
import torch.nn as nn
from fastai.tabular.all import *
from .encoder import Encoder
from .decoder import Decoder


class DecoderEncoder(nn.Module):
    def __init__(self, decoder: Decoder, encoder: Encoder):
        """DecoderEncoder class.

        Args:
            encoder (Encoder): Encoder object.
            decoder (Decoder): Decoder object.
        """
        super(DecoderEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(x)
        decoded_cat, decoded_cont = decoded
        encoded = self.encoder(decoded_cat, decoded_cont)
        return encoded