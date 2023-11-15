import torch
from .nn.decoder import Decoder
from .utils import decode_cats
import numpy as np
import pandas as pd

def generate_data_from_z(
    decoder: Decoder,
    z_limits: np.ndarray,
    cat_names: list[str],
    cont_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    means_y: np.ndarray,
    stds_y: np.ndarray,
    device: torch.device,
    cat_dict: dict[str, int],
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
    cat_preds, cont_preds = decoded
    cat_reduced = decode_cats(cat_preds, cat_dict)
    
    df_cat = pd.DataFrame(cat_reduced.cpu().numpy(), columns=cat_names)
    conts = cont_preds.detach().cpu().numpy() * stds + means
    df_cont = pd.DataFrame(conts, columns=cont_names)
    df = pd.concat([df_cat, df_cont], axis=1)
    return df, points