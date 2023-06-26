import pytorch_lightning as pl
import torch
from torch import nn as nn
from src.models.components import LinearBlock, activation_function, compute_input


class InverseKinematicsHead(pl.LightningModule):
    """
    Policy implementation (imitation learning) to navigate Maze/Habitat environment

        Args:
            emb_dim: Embedding dimension of the input. Should be the size of the output of the
            local backbone.
            n_actions: Number of actions.
            aggregation: Aggregation strategy for anchor and positive codes. See comments.
            add_norm: Add norm as a feature for the locomotion network.
            bn_momentum: Batch norm momentum if we normalize input.
            activation: Activation function
            p_dropout: Dropout probability
    """

    def __init__(self, emb_dim: int, n_actions: int, aggregation: str = 'concat', add_norm: bool = False,
                 n_layers: int = 2, normalize_input: bool = False, bn_momentum: float = 0.1,
                 activation: str = 'relu', p_dropout: int = 0.1):
        super().__init__()
        # Embedding dimension of the model
        self.emb_dim = emb_dim
        self.aggregation = aggregation
        self.add_norm = add_norm
        self.activation = activation_function(activation)
        self.p_dropout = p_dropout

        # Each aggregation strategy may induce a different input dim
        input_dim_map = {
            'concat': 2 * self.emb_dim,  # Classic, we just concat anchor and positive
            'delta': self.emb_dim,  # Model is trained on delta vector positive - anchor
            'relative': 2 * self.emb_dim,  # Model is trained on anchor concat delta vector
        }
        self.input_dim = input_dim_map[self.aggregation]

        if add_norm:
            self.input_dim += 1

        # Build locomotion
        # Takes as input the anchor code + one hot representation of the action
        if normalize_input and bn_momentum > 0.0:
            clf = [nn.BatchNorm1d(self.input_dim, eps=1e-05, momentum=bn_momentum, affine=False)]  # Normalization layer
        else:
            clf = []
        clf.extend([LinearBlock(input_dim=self.input_dim, output_dim=self.input_dim) for _ in range(n_layers)])
        clf.append(LinearBlock(input_dim=self.input_dim, output_dim=n_actions, relu=False))
        self.mlp = nn.Sequential(*clf)

    def forward(self, anchor_codes: torch.Tensor, positive_codes: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Args:
            x: Batch of Images

        Returns:
            Selected action to navigate agent
        """
        # Forward through encoder
        combined = compute_input(anchor_codes, positive_codes, self.aggregation, self.add_norm, 'locomotion')
        x = self.mlp(combined)

        return x

    def predict(self, anchor_codes: torch.Tensor, positive_codes: torch.Tensor):
        """
        Predict action from backbone codes.

        Returns:
            Selected action. 0 for not connected. 1 for stop. 1+ for positive action.
        """
        output = self(anchor_codes, positive_codes)

        # First output indicates connectivity
        not_connected = torch.nn.functional.sigmoid(output[:, 0]) < .5
        pos_action = output[:, 1:].argmax(dim=-1) + 1
        pos_action[not_connected] = 0

        return pos_action