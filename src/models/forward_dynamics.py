from typing import Dict, List, Optional, Tuple, Union

import functools
import pytorch_lightning as pl
import torch
from torch import nn as nn

from src.models.base_model import BaseModel
from src.utils.eval_metrics import aggregate_val_metrics
from src.models.components import activation_function, LinearBlock


class ForwardDynamics(pl.LightningModule):
    """
    Forward Latent Dynamics modeling x_{positive} = f(x_{anchor}, action)

        Args:
            emb_dim: Embedding dimension of the input. Should be the output of the local metric model.
            n_actions: Size of the output space (number of actions).
            n_layers: Number of hidden layers to include in the networks.
            maze_vector_map: Encode actions as 2D directions instead of one hot actions.
            activation: Activation function for the model
            normalize_input: Whether to normalize input before being feed to the model
    """

    def __init__(self, emb_dim: int, n_actions: int, n_layers: int = 2,
                 maze_vector_map: bool = False, activation: str = 'relu', p_dropout: float = 0.0,
                 normalize_input: bool = False):

        super().__init__()
        # Embedding dimension of the model
        self.emb_dim = emb_dim
        # Feed forward all actions but {NON_CONNECTED, STOP} - Assume action zero is FORWARD
        self.n_actions = n_actions - 2
        self.maze_vector_map = maze_vector_map
        self.activation = activation_function(activation)

        # Build forward dynamics
        # Takes as input the anchor code + one hot representation of the action
        transition_dim = emb_dim + (self.n_actions if not self.maze_vector_map else 2)

        if normalize_input:
            regressor = [nn.BatchNorm1d(transition_dim, eps=1e-05, momentum=0.1, affine=False)]  # Normalization layer
        else:
            regressor = [nn.Identity()]

        regressor.extend([LinearBlock(input_dim=transition_dim, output_dim=transition_dim) for _ in range(n_layers)])
        regressor.append(LinearBlock(input_dim=transition_dim, output_dim=emb_dim, relu=False))
        self.transition = nn.Sequential(*regressor)

        # Maze vector map (see self.maze_vector_map)
        self.maze_map = torch.Tensor([
            [0, 0],  # Stop
            [0, 1],  # North
            [0, -1],  # South
            [-1, 0],  # West
            [0, 1],  # East
            [-1, 1],  # Northwest
            [1, 1],  # Northeast
            [-1, -1],  # Southwest
            [1, -1],  # Southeast
        ]).float()

    def encode_actions(self, actions: torch.Tensor):
        if self.maze_vector_map:
            actions_processed = self.maze_map[actions].to(actions.device)
        else:
            actions_processed = torch.nn.functional.one_hot(actions.long(), num_classes=self.n_actions)
        return actions_processed

    def forward(self, x_anchor: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Args:
            x_anchor: Batch of anchor codes.
            actions: Discrete action ids (not one-hot encoded).

        Returns:
            Positive code embedding
        """
        x_anchor = x_anchor.float()
        # Train dynamics
        # Provide forward dynamics with both anchor code and actions
        actions_processed = self.encode_actions(actions)
        x_forward = torch.cat((x_anchor, actions_processed), dim=1)
        x_hat = x_anchor + self.transition(x_forward)

        return x_hat.double()

    def forward_all_actions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward all actions given an anchor x
        Args:
            x: Anchor latent code

        Returns:
            Reconstructed positives tensor
        """
        codes = torch.repeat_interleave(x, repeats=self.n_actions, dim=0)
        actions = torch.tile(torch.arange(self.n_actions), (x.shape[0],)).to(codes.device)
        positives = self(codes, actions)
        return positives.reshape((x.shape[0], self.n_actions, -1))


class ForwardDynamicsHead(BaseModel):
    """
    Forward dynamics model, reconstructs a positive latent code given the current anchor and discrete action

        Args:
            net: Forward dynamics model
            optimizer: functools.partial torch.optim optimizer with all arguments except
            model parameters. Please refer to the Hydra doc for partially initialized modules.
            loss: Loss function of the model
            lr_scheduler_config: lr_scheduler_config dict. Please refer to pytorch lightning doc.
            model parameters. Please refer to the Hydra doc for partially initialized modules.
            log_figure_every_n_epoch: Frequency at which figures should be logged. Ignored.
            backbone_path: Path to trained local_metric checkpoint. Will be used as the backbone.
            noise: Gaussian noise added to input during training.
            step_every: Number of environment batches to forward before calling an optimizer step. Useful when using
            per_env_batches and a "batch" consists of a large number of environment batches

    """

    def __init__(self,
                 net: pl.LightningModule,
                 optimizer: functools.partial,
                 loss: nn.MSELoss,
                 lr_scheduler_config: Dict = None,
                 log_figure_every_n_epoch: int = 1,
                 backbone_path: str = None,
                 noise: float = 0.0,
                 step_every: int = 4):
        super(ForwardDynamicsHead, self).__init__()

        # Save hyperparameters to checkpoint
        self.save_hyperparameters(logger=False)
        # Load model
        self.net = net

        # Loss definition
        self.loss = loss
        self.backbone = None
        # Manual backwards
        self.automatic_optimization = False

    def forward(self, x_anchor: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward Method for FD

        Args:
            x_anchor: Embedding representation of an anchor
            actions: Discrete action ids.

        Returns:
            Reconstructed positive x_positive_hat and params for latent distribution
        """
        # Result is a dict
        result = self.net.forward(x_anchor=x_anchor.float(), actions=actions)
        return result.double()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Main training loop
        """
        # Zero grad
        opt = self.optimizers()
        opt.zero_grad()

        # Single env
        if isinstance(batch, Dict):
            batch = [batch]

        loss_all = torch.tensor(0.0)

        # Traverse batches in random_order
        n_batches = len(batch)
        perm = torch.randperm(n_batches)

        for i, j in enumerate(perm):
            b = batch[j]

            # Compute loss and anchor embedding
            loss, _, metrics = self._shared_loss_computation(b)

            # Log metrics
            stage = 'train'
            log_metrics = {stage + '/' + key: val for key, val in metrics.items()}
            self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            # Backward the loss iteratively to save GPU space
            # This is implicitly like summing all the loss terms
            loss = loss / len(batch)
            self.manual_backward(loss)  # Multiply the average factor for each env
            loss_all += loss.item()

            if (i + 1) % self.hparams.step_every == 0.0 or i == n_batches - 1:
                opt.step()
                opt.zero_grad()

        # Only for reporting
        self.log_dict({"train/loss_avg": loss_all}, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int,
                        dataloader_idx: Optional[int] = None) -> torch.Tensor:
        """
        Validation loop
        """
        # Compute loss
        loss, _, metrics = self._shared_loss_computation(val_batch)
        # Condition to fetch environment properly if number of environments is equal to one
        dataloader_idx = 0 if dataloader_idx is None else dataloader_idx
        # Fetch environment name
        env = self.fetch_environment(stage='val', dataloader_idx=dataloader_idx)
        # Compute loss and acc per epoch
        stage = 'val'
        log_metrics = {stage + '/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        return loss

    def test_step(self, test_batch: Dict[str, torch.Tensor], batch_idx: int,
                  dataloader_idx: Optional[int] = None) -> torch.Tensor:
        """
        Test loop
        """
        # Compute loss
        loss, _, metrics = self._shared_loss_computation(test_batch)
        # Condition to fetch environment properly if number of environments is equal to one
        dataloader_idx = 0 if dataloader_idx is None else dataloader_idx
        # Fetch environment name
        env, prefix = self.get_test_dataloader_env(dataloader_idx)
        # Compute loss and acc per epoch
        stage = prefix
        log_metrics = {stage + '/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        return loss

    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Aggregate validation metrics of all environments
        log_metrics = aggregate_val_metrics(self.trainer.logged_metrics)
        self.log_dict(log_metrics, prog_bar=True, logger=True)

    def test_epoch_end(self, test_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Aggregate test metrics of all environments
        log_metrics = aggregate_val_metrics(self.trainer.logged_metrics)
        self.log_dict(log_metrics, prog_bar=True, logger=True)

    def _shared_loss_computation(self,
                                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute loss for the FD network

        Args:
            batch: Either train or validation batch

        Returns:
            Loss value for given batch and anchors prediction
        """

        anchors, positives = batch['anchors'], batch['positives']

        # Forward through Local metric network
        self.backbone.eval()  # Otherwise batch norm is on
        # Remove contrasting mode from backbone
        self.backbone.contrast = False
        with torch.no_grad():
            # Assume we work directly from images
            anchors, positives, logits = self.backbone(anchors, positives)
            # Compute stops mask to mask out during training
            stop_logits = logits[:, 1:]
            stop_actions = stop_logits.argmax(dim=1)
            stop_mask = stop_actions > 0
            # Compute training actions ignoring NON_CONNECTED and STOP
            train_logits = logits[:, 2:]
            train_actions = train_logits.argmax(dim=1)

        # Forward dynamics + latent codes jiterring
        if self.training and self.hparams.noise > 0:
            anchors += torch.randn_like(anchors) * self.hparams.noise
        output = self.forward(anchors, train_actions)

        # If all samples are stop, mask the loss instead of masking anchors
        if (~stop_mask).sum() == train_actions.size(0):
            # Negate mask
            stop_mask = ~stop_mask
            loss = self.loss(output * stop_mask.unsqueeze(1), positives * stop_mask.unsqueeze(1))
            # Check average norm of latent codes
            norm = torch.linalg.norm(output * stop_mask.unsqueeze(1) - anchors * stop_mask.unsqueeze(1), dim=1).mean()
        # Mask out stops if there are still positive transitions after masking
        else:
            # Filter examples with STOP on it
            output = output[stop_mask]
            anchors = anchors[stop_mask]
            positives = positives[stop_mask]
            loss = self.loss(output, positives)
            # Check average norm
            norm = torch.linalg.norm(output - anchors, dim=1).mean()

        return loss, output, {'loss': loss, 'norm': norm}
