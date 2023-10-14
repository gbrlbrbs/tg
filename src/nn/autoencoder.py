import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        """Autoencoder class. Will return both the output and the encoded tensor.
        
        Args:
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            latent_size (int): Latent size.
        """
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(True))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: torch.Tensor = self.encoder(x)
        y: torch.Tensor = self.decoder(encoded)
        return y, encoded