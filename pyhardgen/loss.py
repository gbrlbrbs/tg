import torch.nn.functional as F
import torch.nn as nn
import torch

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
        

class LossDecoder(nn.Module):
    def __init__(self, cat_dict: dict[str, int]) -> None:
        """Decoder loss.

        Args:
            cat_dict (dict[str, int]): Dictionary of categories.
        """
        super(LossDecoder, self).__init__()
        self.ce = F.cross_entropy
        self.mse = nn.MSELoss()
        self.cat_dict = cat_dict

    def forward(self, preds: tuple[torch.Tensor, torch.Tensor], cat_targets: torch.Tensor, cont_targets: torch.Tensor) -> torch.Tensor:
        cat_preds, cont_preds = preds
        ce, pos = cat_preds.new([0]), 0
        for i, (_, v) in enumerate(self.cat_dict.items()):
            
            ce += self.ce(cat_preds[:, pos:pos+v].cpu(), cat_targets[:, i].cpu().long())
            pos += v
        
        norm_cats = cat_preds.new([len(self.cat_dict)])
        norm_conts = cont_preds.new([cont_targets.size(1)])
        cat_loss = ce / norm_cats
        cont_loss = self.mse(cont_preds.cpu(), cont_targets.cpu()) / norm_conts
        total = cat_loss + cont_loss

        return total / cat_preds.size(0)