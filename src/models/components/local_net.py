from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops.layers.torch import Rearrange
from einops import rearrange

import pytorch_lightning as pl

from src.models.components import activation_function, init_xavier


# class GELU(torch.nn.Module):
#     """
#     Issue with GELU, had to patch it
#     https://github.com/IIGROUP/MANIQA/issues/15
#     """
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return torch.nn.functional.gelu(input)
#
#
# torch.nn.modules.activation.GELU = GELU


###########################
###  Transformer Head   ###
###########################
class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size: int, num_heads: int, sequence_length: int):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # Compute values, queries and keys -- Pass each for a Dense layer with biases
        self.w_values = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size, bias=True)
        self.w_keys = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size, bias=True)
        self.w_queries = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size, bias=True)

        # Projection of concatenated queries, keys and values
        self.fc_out = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size, bias=True)

    def get_attention_weights(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Compute the attention weights.
        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as
            weights = softmax(Q * K^{T} / sqrt(head_size))
        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.
        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.
        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.
        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """

        # Compute Attention Weights
        energy = torch.einsum('bhqd, bhkd -> bhqk', [queries, keys]) / torch.sqrt(
            torch.tensor(self.head_size))  # Out -> (B, H, Q_len, K_len)
        weights = F.softmax(energy, dim=3)  # Out -> (B, H, Q, K)

        return weights

    def apply_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                        torch.Tensor]:
        """Apply the attention.
        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by
            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)
        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.
        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.
        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.
        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads.
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. Also returns attention weights
        """
        batch_size = queries.size(0)
        # Compute Attention weights
        weights = self.get_attention_weights(queries, keys)  # Out -> (B, H, Q_len, K_len)
        # Convex combination with V (key_len = value_len)
        attended_values = torch.einsum('bhqv, bhvd -> bhqd', [weights, values])  # Out -> (B, H, Q_len, D)
        # Permute and reshape attended_values to proper concatenation
        outputs = self.merge_heads(attended_values)  # Out -> (B, Q_len, H*D)

        return outputs, weights

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split the head vectors.
        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.
        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        batch_size = tensor.size(0)
        # Split heads
        s_heads = tensor.reshape(batch_size, self.sequence_length, self.num_heads, -1)  # Out -> (B, S, H, D)
        # Permute dimension
        s_heads = s_heads.permute(0, 2, 1, 3)  # Out -> (B, H, S, D)

        return s_heads

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge the head vectors.
        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.
        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        batch_size = tensor.size(0)
        # Permute heads
        concat_heads = tensor.permute(0, 2, 1, 3)  # Out -> (B, S, H, D)
        # Concat heads
        concat_heads = concat_heads.reshape(batch_size, self.sequence_length, -1)  # Out -> (B, S, H*D)

        return concat_heads

    def forward(self, hidden_states: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Multi-headed attention.
        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by
            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values
            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection
        Here "*" is the matrix multiplication.
        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.
        return_attention Return attention of current layer
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        # Compute Q, K, V and split_heads
        queries = self.split_heads(self.w_queries(hidden_states))  # Out -> (B, H, S, D)
        keys = self.split_heads(self.w_keys(hidden_states))  # Out -> (B, H, S, D)
        values = self.split_heads(self.w_values(hidden_states))  # Out -> (B, H, S, D)

        # Compute attention
        h, weights = self.apply_attention(queries, keys, values)  # Out -> (B, S, H*D)

        if return_attention:
            return weights

        # Compute final projection
        output = self.fc_out(h)  # Out -> (B, S, H*D)

        return output


class PostNormAttentionBlock(nn.Module):
    """
    Inputs:
        embed_dim - Dimensionality of input and attention feature vectors
        hidden_dim - Dimensionality of hidden layer in feed-forward network
                     (usually 2-4x larger than embed_dim)
        num_heads - Number of heads to use in the Multi-Head Attention block
        dropout - Amount of dropout to apply in the feed-forward network
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, sequence_length: int, dropout: float = 0.1):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim // num_heads, num_heads, sequence_length)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_outputs = self.attn(x)
        # print(inp_x.shape)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs


class PreNormAttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, sequence_length: int, dropout: int = 0.1):
        """A decoder layer.
        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim // num_heads, num_heads, sequence_length)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_outputs = self.attn(self.layer_norm_1(x))
        attention_outputs = x + attention_outputs
        outputs = self.linear(self.layer_norm_2(attention_outputs))

        outputs = outputs + attention_outputs
        return outputs


class TransformerHead(nn.Module):
    """
    Transformer Encoder predictor head for Local Metric.
        Args:
            seq_len: Lenght of the sequence i.e. k
            embed_dim: Dimensionality of the input feature vectors to the Transformer Encoder
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks within the Transformer
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer i.e. Encoder blocks
            block: Type of attention block
            dropout: Amount of dropout to apply in the feed-forward network and on the input encoding
            use_cls: Return the CLS token in the forward method or the last position (anchor)
    """

    def __init__(self, seq_len: int = 4, embed_dim: int = 8, hidden_dim: int = 32, num_heads: int = 2,
                 num_layers: int = 4, block: int = 'prenorm', dropout: float = 0.1, use_cls: bool = True):
        super().__init__()

        assert embed_dim % num_heads == 0, "Make sure number of heads is factor of embedding dimension"

        self.use_cls = use_cls

        # Adding the cls token to the sequnence
        self.sequence_length = 1 + seq_len
        # Layers/Networks
        if block == 'prenorm':
            self.blocks = nn.ModuleList([
                PreNormAttentionBlock(embed_dim, hidden_dim, num_heads, self.sequence_length, dropout=dropout)
                for _ in range(num_layers)])
        else:
            self.blocks = nn.ModuleList([
                PostNormAttentionBlock(embed_dim, hidden_dim, num_heads, self.sequence_length, dropout=dropout)
                for _ in range(num_layers)])
        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_length, embed_dim))

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input by adding CLS token and add positional embeddings to sequence.
        Args:
            x: Input sequence
        Returns:
            Processed input sequence
        """
        # Preprocess input
        B, S, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :S + 1, :]

        # Add dropout and feed to transformer
        x = self.dropout(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method Transformer Encoder.
        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, Sequence, emb_dim)`)
            The input tensor containing the embeddings.
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, emb_dim)`)
            A tensor containing the output from the CLS token or the anchor embedding.
        """
        # Process Input
        x = self.prepare_input(x)

        # Pass it through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Return CLS token
        if self.use_cls:
            x = x[:, 0, :]
        # Return Anchor token which is the last embedding
        else:
            x = x[:, -1, :].unsqueeze(1)

        # Add one extra dim for compatibility with Habitat backbone
        x = rearrange(x, 'b d -> b 1 d')
        return x

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_input(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_input(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class PositionalEncoding(nn.Module):
    """
    Transformer positional encoding
        Args:
            d_model: Input embedding dimension which is equal to emb_dim.
            dropout: Dropout percentage.
            max_len: Max length of embedding dimension
    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


###########################
###  Conv1D Head Maze   ###
###########################
class Conv1DHead(nn.Module):
    """
    Convolutional predictor head for the Local Metric
        Args:
            emb_dim: Embedding dimension of the output
            k: Context length i.e. sequence length
            activation: Activation function used for the model
    """

    def __init__(self, emb_dim: int, k: int, activation: object):
        super().__init__()
        self.emb_dim = emb_dim
        self.k = k
        self.activation = activation

        self.head = nn.Sequential(
            nn.Conv1d(in_channels=self.k, out_channels=self.k * 4, kernel_size=1, stride=1, padding='same'),
            self.activation,
            nn.Conv1d(in_channels=self.k * 4, out_channels=self.k * 8, kernel_size=1, stride=1, padding='same'),
            self.activation,
            nn.Conv1d(in_channels=self.k * 8, out_channels=self.k * 16, kernel_size=1, stride=1, padding='same'),
            self.activation,
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=int(self.k * 16 * self.emb_dim / 2), out_features=512, bias=True),
            self.activation,
            nn.Linear(in_features=512, out_features=512, bias=True),
            self.activation,
            nn.Linear(in_features=512, out_features=self.emb_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project codes produced by backbone into predictor space
        Args:
            x: Sequence of embeddings
        Returns:
            Single embedding for a given sequence
        """
        return self.head(x)


####################
###  Maze Model  ###
####################
class LocalMetricMaze(pl.LightningModule):
    """
    Local metric network implementation for a Maze Environments.
        Args:
            emb_dim: Embedding dimension of the output
            k: Context length i.e. number of images in panorama.
            predictor: Options are {conv1d, transformer, mlp}
            num_blocks: Number of transformer blocks
            dropout: (NOT USED) - dropout probability
            num_heads: Number of transformers head per block
            use_cls: Whether to use CLS token as final embedding or actual anchor embedding
            norm: Normalize output to norm 1.
            activation: Activation function used to for the model, options are:
            [sigmoid, relu, tanh, leaky_relu, elu, mish]
            normalize_output: Standard scale the output with a batchnorm layer.
            bn_momentum: Batch norm momentum. Set to 0 for no batch norm.
    """

    def __init__(self, emb_dim: int,
                 k: int,
                 predictor: str = "conv1d",
                 num_blocks: int = 2,
                 dropout: float = 0.0,
                 num_heads: int = 2,
                 use_cls: bool = True,
                 norm: bool = False,
                 activation: str = 'relu',
                 normalize_output: bool = False,
                 bn_momentum: float = .1):
        super().__init__()
        # Embedding dimension of the model
        self.emb_dim = emb_dim
        self.k = k
        self.predictor = predictor
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_cls = use_cls
        self.norm = norm
        self.normalize_output = normalize_output
        self.bn_momentum = bn_momentum
        self.bn = self.bn_momentum > 0.0

        assert self.predictor in {'conv1d', 'transformer', 'mlp'}, \
            "Please select one valid predictor -> {conv1d, transformer, mlp}"

        self.activation = activation_function(activation)

        # Assume 64 x 64 grayscale images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(64, 32, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=self.bn_momentum, affine=True) if self.bn else nn.Identity(),
            self.activation,
        )

        # Metric head
        self.head = nn.Sequential(
            nn.Linear(in_features=128, out_features=128, bias=True),
            self.activation,
            nn.Linear(in_features=128, out_features=128, bias=True),
            self.activation,
            nn.Linear(in_features=128, out_features=self.emb_dim, bias=True)
        )

        if self.normalize_output and self.bn:
            scale_layer = nn.BatchNorm1d(self.emb_dim, eps=1e-05, momentum=self.bn_momentum, affine=False)
        else:
            scale_layer = nn.Identity()

        # Predictor for sequence
        if predictor == "conv1d":
            self.predictor = nn.Sequential(
                Conv1DHead(emb_dim=self.emb_dim, k=self.k, activation=self.activation),
                scale_layer
            )
        elif predictor == 'transformer':
            self.predictor = nn.Sequential(
                TransformerHead(seq_len=self.k,
                                num_layers=self.num_blocks,
                                dropout=self.dropout,
                                num_heads=self.num_heads,
                                embed_dim=self.emb_dim,
                                hidden_dim=self.emb_dim * 4,
                                block="prenorm",
                                use_cls=self.use_cls),
                scale_layer
            )
        elif predictor == 'mlp':
            # Feed forward MLP + Mean layer and scale
            self.predictor = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=self.emb_dim * self.k, out_features=self.emb_dim * 4, bias=True),
                self.activation,
                nn.Linear(in_features=self.emb_dim * 4, out_features=self.emb_dim * 4, bias=True),
                self.activation,
                nn.Linear(in_features=self.emb_dim * 4, out_features=self.emb_dim, bias=True),
                scale_layer
            )

    def forward(self, x_anchor: torch.Tensor,
                x_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Local metric forward method for the Maze Environment.
        Args:
            x_anchor: Batch of sequence of images (anchors) [B, S, C, H, W]
            x_pos: Batch of positives for anchors [B, C, H, W]
        Returns:
            Embedding representation of anchors (individually) [B, D]
            Embedding representation of positives - is returned only if positives are provided [B, D]
            Embedding representation of sequence of anchors (combined) [B, K, D]
        """
        # Set embedding of positives to None in case it is not provided
        p_pos = None
        # Flatten first dimension for anchors [B, N, S, C, H, W] -> [B * N * S, C, H, W]
        b, n, s, c, h, w = x_anchor.size()
        # Forward through encoder .reshape((b * n * s, s, c, w, h))
        z_anchor = self.encoder(x_anchor.view(-1, c, h, w))
        # Forward though linear projector - ignore n dimension for now
        h_anchor = self.head(z_anchor.view(b * n, s, 128))
        if self.norm:
            h_anchor = torch.nn.functional.normalize(h_anchor, p=2.0, dim=-1)
        # Forward though predictor and concat sequence in one batch - retrieve n dimension
        p_anchor = self.predictor(h_anchor).view(b, n, -1).squeeze()
        if self.norm:
            p_anchor = torch.nn.functional.normalize(p_anchor, p=2.0, dim=-1)

        # If positives are provided forward those
        if x_pos is not None:
            # Extract shapes again as x_pos can have a
            b, n, s, c, h, w = x_pos.size()
            z_pos = self.encoder(x_pos.view(-1, c, h, w))
            h_pos = self.head(z_pos.view(b * n, s, 128))
            if self.norm:
                h_pos = torch.nn.functional.normalize(h_pos, p=2.0, dim=-1)
            # Forward though predictor and concat sequence in one batch
            p_pos = self.predictor(h_pos).view(b, n, -1).squeeze()
            if self.norm:
                p_pos = torch.nn.functional.normalize(p_pos, p=2.0, dim=-1)

        return p_anchor, p_pos


#############################
###  Conv1D Head Habitat  ###
#############################
class Conv1DHeadHabitat(nn.Module):
    """
    Convolutional predictor head for the Local Metric Habitat/Jackal
        Args:
            emb_dim: Embedding dimension of the output
            k: Context length i.e. sequence length
            activation: Activation function used for the model
    """

    def __init__(self, emb_dim: int, k: int, activation: object, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.k = k
        self.activation = activation
        self.dropout = dropout

        out_features = 256 if self.emb_dim <= 256 else self.emb_dim

        self.head = nn.Sequential(
            nn.Conv1d(in_channels=self.k, out_channels=8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(8),
            self.activation,
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            self.activation,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            self.activation,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Flatten(),
            nn.Dropout(p=self.dropout),
            Rearrange('b d -> b 1 d'),
            nn.Linear(in_features=512, out_features=out_features, bias=True),
            self.activation,
            nn.Linear(in_features=out_features, out_features=self.emb_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project codes produced by backbone into predictor space
        Args:
            x: Sequence of embeddings
        Returns:
            Single embedding for a given sequence
        """
        return self.head(x)


class LocalMetricHabitat(pl.LightningModule):
    """
    Local metric network implementation from Habitat/Jackal Environment.
        Args:
            emb_dim: Embedding dimension of the output
            k: Context length i.e. number of images in panorama.
            n_positives: Number of positives in the input
            encoder_type: (NOT USED) which encoder to use {default, resnet}
            predictor: Options are {conv1d, transformer, mlp}
            num_blocks: Number of transformer blocks
            dropout: (NOT USED) - dropout probability
            num_heads: Number of transformers head per block
            use_cls: Whether to use CLS token as final embedding or actual anchor embedding
            activation: Activation function used to for the model, options are:
            [sigmoid, relu, tanh, leaky_relu, elu, mish]
            normalize_output: Standard scale the output with a batchnorm layer.
            bn_momentum: Batch norm momentum. Set to 0 for no batch norm.
    """

    def __init__(self, emb_dim: int,
                 k: int,
                 n_positives: Union[int, str],
                 encoder_type: str = "resnet",
                 predictor: str = "conv1d",
                 num_blocks: int = 2,
                 dropout: float = 0.0,
                 num_heads: int = 2,
                 use_cls: bool = True,
                 activation: str = 'relu',
                 normalize_output: bool = False,
                 bn_momentum: float = .1):
        super().__init__()
        # Embedding dimension of the model
        self.emb_dim = emb_dim
        self.k = k
        self.predictor = predictor
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_cls = use_cls
        self.normalize_output = normalize_output
        self.bn_momentum = bn_momentum
        self.encoder_type = encoder_type
        self.bn = self.bn_momentum > 0.0

        out_features = 256 if self.emb_dim <= 256 else self.emb_dim

        # Define number of positives
        if n_positives == 'temporal':
            self.n_positives = 1
        elif isinstance(n_positives, int) and n_positives > 0:
            self.n_positives = n_positives
        else:
            raise ValueError('n_positives should be "temporal" or a positive int')

        # Assess predictor type
        assert self.predictor in {'conv1d', 'transformer', 'mlp'}, \
            "Please select one valid predictor -> {conv1d, transformer, mlp}"
        assert activation in {'sigmoid', 'relu', 'tanh', 'leaky_relu', 'elu', 'mish'}, \
            "Please select one valid predictor -> {sigmoid, relu, tanh, leaky_relu, elu, mish}"

        self.activation = activation_function(activation)

        # Assume 96 x 96 RGB images
        # resnet = models.resnet18(pretrained=False)  # Download from torch hub
        resnet = torch.load('models/resnet18.pt')  # Load local checkpoint

        # Read as b = batch size, n = number of positives, s = sequence dimension
        # c = number of channels, h = image height, w = image width
        encoder = [Rearrange('b n s c h w -> (b n s) c h w')]
        encoder.extend(list(resnet.children())[0:9])
        encoder.extend([Rearrange('(b n s) c h w -> (b n) s (c h w)', n=self.n_positives, s=self.k)])
        self.encoder = nn.Sequential(*encoder)

        if self.normalize_output and self.bn:
            self.scale_layer = nn.BatchNorm1d(self.emb_dim, eps=1e-05, momentum=self.bn_momentum, affine=False)
        else:
            self.scale_layer = nn.Identity()

        # Sequence predictor
        if predictor == "conv1d":
            # Predictor Head for Conv1d
            predictor = [
                Conv1DHeadHabitat(emb_dim=self.emb_dim, k=self.k, activation=self.activation, dropout=self.dropout),
                # d stands for embedding dimension
                Rearrange('(b n) s d -> b n (s d)', n=self.n_positives),
                self.scale_layer
            ]
            self.predictor = nn.Sequential(*predictor)
        elif predictor == 'transformer':
            # Predictor Head for Transformer
            self.predictor = nn.Sequential(
                TransformerHead(seq_len=self.k,
                                num_layers=self.num_blocks,
                                dropout=self.dropout,
                                num_heads=self.num_heads,
                                embed_dim=512,
                                hidden_dim=2048,
                                block="prenorm",
                                use_cls=self.use_cls),
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=512, out_features=out_features, bias=True),
                self.activation,
                nn.Linear(in_features=out_features, out_features=self.emb_dim, bias=True),
                Rearrange('(b n) s d -> b n (s d)', n=self.n_positives),
                self.scale_layer
            )
        elif predictor == 'mlp':
            # Feed forward MLP + Mean layer and scale
            self.predictor = nn.Sequential(
                Rearrange('(b n) s d -> (b n) (s d)', n=self.n_positives),
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=512 * self.k, out_features=512, bias=True),
                self.activation,
                nn.Linear(in_features=512, out_features=out_features, bias=True),
                self.activation,
                nn.Linear(in_features=out_features, out_features=self.emb_dim, bias=True),
                Rearrange('(b n) d -> b n d', n=self.n_positives),
                self.scale_layer
            )

        # Initialize all models weights with xavier
        self.encoder.apply(init_xavier)
        self.predictor.apply(init_xavier)

    def forward(self, x_anchor: torch.Tensor,
                x_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Local metric forward method for the Maze Environment.
        Args:
            x_anchor: Batch of sequence of images (anchors) [B, S, C, H, W]
            x_pos: Batch of positives for anchors [B, C, H, W]
        Returns:
            Embedding representation of anchors (individually)
            Embedding representation of positives - is returned only if positives are provided
            Embedding representation of sequence of anchors (combined)
        """
        # Set embedding of positives to None in case it is not provided
        p_pos = None
        # Forward through encoder
        z_anchor = self.encoder(x_anchor)
        # Forward though predictor and concat sequence in one batch - retrieve n dimension
        p_anchor = self.predictor(z_anchor).squeeze()

        # If positives are provided forward those
        if x_pos is not None:
            z_pos = self.encoder(x_pos)
            # Forward though predictor and concat sequence in one batch
            p_pos = self.predictor(z_pos).squeeze()

        return p_anchor, p_pos


