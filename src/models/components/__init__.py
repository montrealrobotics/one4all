import torch
from torch import nn as nn

import pytorch_lightning as pl


class LinearBlock(pl.LightningModule):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, residual: bool = False, relu: bool = True):
        super().__init__()
        # Skip connection
        self.residual = residual

        # Embedding dimension of the model
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        if relu:
            self.activation_1 = nn.ReLU()
        else:
            self.activation_1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_1(x)
        out = self.activation_1(out)

        if self.residual:
            out = out + x

        return out


def activation_function(activation: str):
    if activation == 'sigmoid':
        act_func = nn.Sigmoid()
    elif activation == 'relu':
        act_func = nn.ReLU()
    elif activation == 'tanh':
        act_func = nn.Tanh()
    elif activation == 'leaky_relu':
        act_func = nn.LeakyReLU()
    elif activation == 'elu':
        act_func = nn.ELU()
    elif activation == 'mish':
        act_func = nn.Mish()
    else:
        raise (NotImplementedError,
               'Activation Function Not Implemented. Needs to be one of: [sigmoid, relu, tanh, leaky_relu, elu, mish]')

    return act_func


# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_xavier(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def compute_input(anchor_codes: torch.Tensor, positive_codes: torch.Tensor,
                  aggregation: str, add_norm: bool, arch: str = 'locomotion') -> torch.Tensor:
    """
    Concat anchor codes based on specific aggregation
    Args:
        anchor_codes: Anchor codes
        positive_codes: Positive codes
        aggregation: Type of aggregation
        add_norm: Whether to add norm or not
        arch: Type of architecture, options are [locomotion, geodesic]

    Returns:
        Concatenated tensors
    """
    if aggregation == 'concat':
        result = torch.cat((anchor_codes, positive_codes), dim=1)
    elif aggregation == 'delta':
        # Delta vector
        result = positive_codes - anchor_codes
    elif aggregation == 'relative':
        delta = positive_codes - anchor_codes
        if arch == 'locomotion':
            result = torch.cat((anchor_codes, delta), dim=1)
        elif arch == 'geodesic':
            result = torch.cat((anchor_codes, positive_codes, delta), dim=1)
        else:
            raise ValueError('Wrong architecture, specify either geodesic or locomotion.')
    else:
        raise ValueError('Wrong aggregation value.')

    if add_norm:
        norm = torch.linalg.norm(positive_codes - anchor_codes, dim=1).reshape((-1, 1))
        result = torch.cat((result, norm), dim=1)

    return result
