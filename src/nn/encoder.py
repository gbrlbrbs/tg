import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        """Encoder class.

        Args:
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            latent_size (int): Latent size.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        return encoded