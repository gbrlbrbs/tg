import torch
from .nn.decoder import Decoder
import numpy as np
import pandas as pd

def generate_data_from_z(
    decoder: Decoder,
    z_limits: np.ndarray,
    columns: list[str],
    n_samples: int = 100,
) -> pd.DataFrame:
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
    data: torch.Tensor = decoder(torch.from_numpy(points).float())
    df = pd.DataFrame(data.detach().numpy(), columns=columns)
    return df