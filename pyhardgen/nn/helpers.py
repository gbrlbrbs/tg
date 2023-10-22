import torch.nn as nn
from ..config import model

def match_activation(name: str):
    match name.lower():
        case 'relu':
            return nn.ReLU(True)
        case 'tanh':
            return nn.Tanh()
        case 'sigmoid':
            return nn.Sigmoid()
        case _:
            raise ValueError(f'Activation {name} not implemented')