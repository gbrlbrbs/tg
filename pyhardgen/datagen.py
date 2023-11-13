import torch
from .nn.decoder import Decoder
import numpy as np
import pandas as pd

def generate_data_from_z(
    decoder: Decoder,
    z_limits: np.ndarray,
    cat_names: list[str],
    cont_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
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
    decoded = decoder(torch.tensor(points, dtype=torch.float64).to(device))
    cat_preds, cont_preds = decoded
    cat_reduced = torch.zeros((n_samples, len(cat_names)), dtype=torch.long)
    pos = 0
    for i, (_, v) in enumerate(cat_dict.items()):
        cat_reduced[:, i] = torch.argmax(cat_preds[:, pos:pos + v], dim=1)
        pos += v
    
    df_cat = pd.DataFrame(cat_reduced.cpu().numpy(), columns=cat_names)
    conts = cont_preds.detach().cpu().numpy() * stds + means
    df_cont = pd.DataFrame(conts, columns=cont_names)
    df = pd.concat([df_cat, df_cont], axis=1)
    return df, points