import torch.nn.functional as F

def match_loss(loss_str: str):
    """Return a loss function.

    Args:
        loss_str (str): Loss function name.

    Returns:
        function: Loss function.
    """
    match loss_str:
        case 'mse':
            return F.mse_loss
        case 'l1':
            return F.l1_loss
        case 'nll':
            return F.nll_loss
        case 'cross_entropy':
            return F.cross_entropy
        case _:
            raise ValueError(f'Invalid loss function: {loss_str}')