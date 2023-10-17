import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, output_size: int):
        """Decoder class.
        
        Args:
            latent_size (int): Latent size.
            hidden_size (int): Hidden size.
            output_size (int): Output size.
        """
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(x)
        return decoded