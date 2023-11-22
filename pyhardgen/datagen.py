import torch
from .nn.decoder import Decoder
from .utils import decode_cats
import numpy as np
import pandas as pd

def generate_data_from_z(
    decoder: Decoder,
    z_limits: np.ndarray,
    measures: list[str],
    means_y: np.ndarray,
    stds_y: np.ndarray,
    device: torch.device,
    n_samples: int = 100,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data from a decoder.

    Args:
        decoder (`Decoder`): Decoder.
        z_limits (`np.ndarray`): Limits for z. 2x2 array [z1, z2]
        columns (`list[str]`): Columns for the dataframe.
        n_samples (int, optional): Number of samples. Defaults to 100.
    
    Returns:
        `pd.DataFrame`: Generated data.
    """
    points = np.random.uniform(z_limits[0], z_limits[1], size=(n_samples, 2))
    norm_points = (points - means_y) / stds_y
    decoded = decoder(torch.tensor(norm_points, dtype=torch.float64).to(device))
    
    df = pd.DataFrame(decoded.detach().cpu().numpy(), columns=measures)
    return df, points