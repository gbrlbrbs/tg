import torch

def decode_cats(decoded_cats: torch.Tensor, cat_dict: dict[str, int]) -> torch.Tensor:
    """Decode categorical values.

    Args:
        decoded_cats (torch.Tensor): Decoded categorical values.
        cat_dict (dict[str, int]): Dictionary of categories.

    Returns:
        torch.Tensor: Decoded categorical values.
    """
    decoded: torch.Tensor = torch.zeros((decoded_cats.size(0), len(cat_dict)), dtype=torch.long).to(decoded_cats.device)
    pos = 0
    for i, (_, v) in enumerate(cat_dict.items()):
        decoded[:, i] = decoded_cats[:, pos:pos+v].argmax(1)
        pos += v
    return decoded