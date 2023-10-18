import torch
from .nn.decoder import Decoder
import numpy as np

def generate_data_from_z(
    decoder: Decoder,
    z_limits: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    """Generate data from a decoder.

    Args:
        decoder (`Decoder`): Decoder.
        z_limits (`np.ndarray`): Limits for z. 2x2 array [z1, z2]
        n_samples (int, optional): Number of samples. Defaults to 100.
    
    Returns:
        np.ndarray: Generated data.
    """
    points = np.random.uniform(z_limits[0], z_limits[1], size=(n_samples, 2))
    data: torch.Tensor = decoder(torch.from_numpy(points).float())
    return data.detach().numpy()