import torch
import torch.nn as nn

import pytorch_lightning as pl
from src.models.components import activation_function, LinearBlock, compute_input


class GeodesicBackbone(pl.LightningModule):
    """
    Geodesic regressor backbone to chain with the local backbone

    Args:
        emb_dim: Embedding dimension of the output
        n_layers: Number of layers to use for this model
        activation: Activation function
        aggregation: Type of aggregation
        add_norm: Whether to add or not the norm between two latent codes to the input
        p_dropout: Dropout probability
        normalize_input: whether to normalize codes before feeding it into the model or not
    """

    def __init__(self, emb_dim: int = 2, n_layers: int = 4, activation: str = 'relu',
                 p_dropout: float = 0.0, normalize_input: bool = False):
        super().__init__()

        # Embedding dimension of the model
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.activation = activation_function(activation)
        self.p_dropout = p_dropout

        if normalize_input:
            arch = [nn.BatchNorm1d(emb_dim, eps=1e-05, momentum=0.1, affine=False)]  # Normalization layer
        else:
            arch = [nn.Identity()]

        # Architecture
        residual = True
        arch += [nn.Dropout(p=self.p_dropout)]
        arch += [LinearBlock(input_dim=emb_dim, output_dim=emb_dim, residual=False)]
        arch += [LinearBlock(input_dim=emb_dim, output_dim=emb_dim, residual=residual) for _ in range(self.n_layers)]
        arch += [LinearBlock(input_dim=emb_dim, output_dim=emb_dim, residual=False, relu=False)]

        # Metric head
        self.head = nn.Sequential(
            *arch
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Geodesic regressor's backbone forward method

        Args:
            x: Batch of latent codes in local backbone space.

        Returns:
            Geodesic latent codes
        """
        return self.head(x)


class RegressionHead(pl.LightningModule):
    """
    Regression head to estimate heuristic for geodesic regressor - this transforms geodesic regressor or (global) in a
    `quasi metric` as it is not symmetric.

    Args:
        emb_dim: Embedding dimension of the output
        aggregation: Type of aggregation
        add_norm: Whether to add or not the norm between two latent codes to the input
        out_activation: Activation function used as the output to guarantee output is in R^+
        activation: Activation function
        p_dropout: Dropout probability
    """

    def __init__(self, emb_dim: int = 2, aggregation: str = 'concat',
                 add_norm: bool = False, out_activation: str = 'elu',
                 activation: str = 'relu', p_dropout: float = 0.0):
        super().__init__()

        # Embedding dimension of the model
        self.emb_dim = emb_dim
        self.aggregation = aggregation
        self.add_norm = add_norm
        self.activation = activation_function(activation)
        self.p_dropout = p_dropout
        # Pick out activation to guarantee out >= 0
        if out_activation == 'elu':
            self.out_activation = nn.ELU()
        else:
            raise NotImplementedError("Select valid activation for the output -- ELU")

        # Each aggregation strategy may induce a different input dim
        input_dim_map = {
            'concat': 2 * self.emb_dim,  # Classic, we just concat anchor and positive
            'delta': self.emb_dim,  # Model is trained on delta vector positive - anchor
            'relative': 3 * self.emb_dim,  # Model is trained on anchor concat positive concat delta vector
        }

        self.input_dim = input_dim_map[self.aggregation]

        if add_norm:
            self.input_dim += 1

        self.head = nn.Sequential(
            nn.Dropout(p=self.p_dropout),
            nn.Linear(in_features=self.input_dim, out_features=self.input_dim, bias=True),
            self.activation,
            nn.Linear(in_features=self.input_dim, out_features=emb_dim, bias=True),
            self.activation,
            nn.Linear(in_features=emb_dim, out_features=1, bias=True)
        )

    def forward(self, anchor_codes: torch.Tensor, positive_codes: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Args:
            anchor_codes: geodesic anchor code
            positive_codes: geodesic positive code

        Returns:
            Geodesic distance estimate between anchor and positive
        """
        # Forward through encoder
        combined = compute_input(anchor_codes, positive_codes, self.aggregation, self.add_norm, 'geodesic')
        x = self.head(combined)

        # replaced torch.exp(sd) with ELU plus to improve numerical stability
        # added epsilon to avoid zero scale
        return self.out_activation(x) + 1.0

    def forward_all_codes(self, x: torch.Tensor, y: torch.Tensor, deployment: bool = False) -> torch.Tensor:
        """
        Forward one anchor against all other anchors in dataset
        Args:
            x: Anchor tensor
            y: All other anchors in dataset
            deployment: whether we are forwarding codes during deployment or training

        Returns:
            Regression heuristic
        """
        if deployment:
            codes = torch.repeat_interleave(y, repeats=x.size(0), dim=0)
            dist = self(x, codes)
        else:
            codes = torch.repeat_interleave(x, repeats=y.size(0), dim=0)
            dist = self(codes, y)
        return dist.squeeze()
