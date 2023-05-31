from typing import Dict, List, Tuple, Optional

import functools

import matplotlib.pyplot as plt

import torch

from src.models.base_model import BaseModel
from src.models.components.geodesic_nets import GeodesicBackbone, RegressionHead
from src.utils import fig2img, get_logger
from src.utils.visualization import plot_2d_latent, plot_connectivity_graph
from src.utils.eval_metrics import a_star_eval_global

log = get_logger(__name__)


class GeodesicRegressor(BaseModel):
    """
    Geodesic regressor to approximate shortest path length between two latent codes

        Args:
            net: Geogedic regressor's encoder
            head: Regression head to regress shortest path length
            optimizer: functools.partial torch.optim optimizer with all arguments except
            model parameters. Please refer to the Hydra doc for partially initialized modules.
            local_metric_path: Path to trained local_metric checkpoint.
            n_edges: int, number of edges in the local metric debug visualization.
            lr_scheduler_config: lr_scheduler_config dict. Please refer to pytorch lightning doc.
            model parameters. Please refer to the Hydra doc for partially initialized modules.
            log_figure_every_n_epoch: Frequency at which figures should be logged. The counter
            only increases when validation_epoch_end is called.
            noise: Amount of noise used to jitter the latent codes and mprove generalization
            freeze_backbone: Flag to freeze local metric backbone.
    """

    def __init__(self,
                 net: GeodesicBackbone,
                 head: RegressionHead,
                 optimizer: functools.partial,
                 backbone_path: str,
                 n_edges: int = 1000,
                 lr_scheduler_config: Dict = None,
                 log_figure_every_n_epoch: int = 1,
                 noise: float = 1e-1,
                 freeze_backbone: bool = True):
        super(GeodesicRegressor, self).__init__()

        # Save hyperparameters to checkpoint
        self.save_hyperparameters(logger=False)

        # Select model
        self.net = net
        self.head = head

        # Loss
        self.loss = torch.nn.MSELoss()
        self.backbone = None
        self.has_logged_debug_local = False
        self.freeze_backbone = freeze_backbone

    def on_fit_start(self) -> None:
        """
        Validate val and test environments are exactly one
        """
        # Fetch val and test environments
        test_envs = self.trainer.datamodule.hparams.test_environments
        val_envs = self.trainer.datamodule.hparams.val_environments
        assert len(test_envs) == 1, log.error("Please set the number of Test environments equal to one.")
        assert len(val_envs) == 1, log.error("Please set the number of environments equal to one.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Geodesic regressor forward method

        Args:
            x: Batch of Images

        Returns:
            Embedding representation of given image.
        """
        if self.backbone is not None:
            with torch.inference_mode():
                self.backbone.contrast = False
                x, _, _ = self.backbone(x)

        # Map tensor to float as model weights are in float
        x = x.float()
        # Jitter local codes for 'robustness'
        if self.training and self.hparams.noise > 0:
            x += torch.randn_like(x) * self.hparams.noise

        # Forward through Geodesic regressor network
        h = self.net.forward(x)

        return h.double()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Main training loop
        """
        # Compute loss and anchor embedding
        loss, h_anchor = self._shared_loss_computation(batch)

        # Log loss per epoch and RMSE
        log_metrics = {'train/loss': loss, 'train/rmse': torch.sqrt(loss)}
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'anchors': h_anchor, 'anchors_id': batch['anchors_id']}

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int,
                        dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Validation loop
        """
        # Compute loss and anchor embedding
        loss, h_anchor = self._shared_loss_computation(val_batch)

        # Log loss per epoch and RMSE
        log_metrics = {'val/loss': loss, 'val/rmse': torch.sqrt(loss)}
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        return {'anchors': h_anchor, 'gt_anchors': val_batch['gt_anchors'], 'anchors_id': val_batch['anchors_id']}

    def test_step(self, test_batch: Dict[str, torch.Tensor], batch_idx: int,
                  dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Validation loop
        """
        # Compute loss and anchor embedding
        loss, h_anchor = self._shared_loss_computation(test_batch)

        # Log loss per epoch and RMSE
        log_metrics = {'test/loss': loss, 'test/rmse': torch.sqrt(loss)}
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        return {'anchors': h_anchor, 'gt_anchors': test_batch['gt_anchors'], 'anchors_id': test_batch['anchors_id']}

    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, torch.Tensor]], stage: str = 'val') -> None:
        """Report various metrics over the whole validation/test split."""
        # Fetch dataloader and environment
        if stage == 'val':
            dataloader = self.trainer.val_dataloaders[0]
        else:
            dataloader = self.trainer.test_dataloaders[0]
        env = dataloader.dataset.experiment
        stage = dataloader.dataset.split_raw
        # Plot embedding and compute more graph metrics
        if self.log_figures():
            # Extract predictions and gt
            pred = torch.cat([x["anchors"] for x in validation_step_outputs])
            target = torch.cat([x["gt_anchors"] for x in validation_step_outputs]).squeeze().cpu().numpy()
            ids = torch.cat([x["anchors_id"] for x in validation_step_outputs]).squeeze().cpu().numpy()

            # Reorder pred and targets
            # We cannot assume the validation set is ordered
            mask = ids.argsort()
            pred = pred[mask]
            target = target[mask]

            # Offline graph evaluation - Note: Geodesic head is trained only over one environment.
            # Make sure we embedded all the dataset. This check solves issues with the lightning sanity check
            graph = dataloader.dataset.graph
            dist = dataloader.dataset.geodesics_len
            if graph.number_of_nodes() == pred.shape[0]:
                rel_avg_path_len, avg_access_rate, _ = a_star_eval_global(graph, 50, global_codes=pred,
                                                                          regression_head=self.head, dijkstra_dist=dist)
                self.log('{}/rel_avg_path_len'.format(stage), rel_avg_path_len, prog_bar=True, logger=True)
                self.log('{}/avg_access_rate'.format(stage), avg_access_rate, prog_bar=True, logger=True)

            pred = pred.cpu().numpy()
            fig = plot_2d_latent(pred, target[:, :2].sum(axis=1))  # Pick ground truth of last target
            current_epoch = self.trainer.current_epoch
            if self.logger is not None:
                # Log geodesic embeddings figure
                self.log_fig2image(fig, f'{stage}_{env}_geodesic_embedding_epoch_{current_epoch}')

        # Plot local connectivity graph we are working with
        # Do this only once. Useful for debugging
        if not self.has_logged_debug_local:
            # Plot Position graph
            graph = dataloader.dataset.graph
            # Plot graph figures
            figures = list()
            for graph_type in ['pos', 'rot']:
                # Plot rotation connectivity graph only if in SE(2) environments
                if graph_type == 'rot' and not (dataloader.dataset.env_type in {'habitat', 'jackal'}):
                    figure = None
                else:
                    figure = plot_connectivity_graph(graph, environment=dataloader.dataset.env_type,
                                                     n_edges=self.hparams.n_edges, plot_type=graph_type)
                figures.append(figure)
            current_epoch = self.trainer.current_epoch
            if self.logger is not None:
                # Log derived connectivity for translation and rotations
                self.log_fig2image(figures[0], f'{stage}_{env}_pos_connectivity_local_metric_{current_epoch}')
                self.log_fig2image(figures[1], f'{stage}_{env}_rot_connectivity_local_metric_{current_epoch}')
            self.has_logged_debug_local = True

    def test_epoch_end(self, test_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        # To enable test figures logging
        self.figure_counter = 0
        self.hparams.log_figure_every_n_epoch = 1
        # End of epoch evaluation
        self.validation_epoch_end(test_step_outputs, stage='test')

    def _shared_loss_computation(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract datamodule
        anchor, pos, cost = batch['anchors'], batch['positives'], batch['geodesics_len']
        # Compute a mask where True means infinity and will be removed from computation graph
        inf_mask = torch.isinf(cost)

        # Forward method
        h_anchor, h_positive = self.forward(anchor), self.forward(pos)
        # Compute heuristic
        heuristic = self.head(h_anchor.float(), h_positive.float()).squeeze().double()

        # Match L2 norm to the Dijkstra cost - Zero the values that are inf in geodesic target
        loss = self.loss(heuristic * ~inf_mask, cost.masked_fill_(inf_mask, 0))

        return loss, h_anchor

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int,
                     dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Prediction step, forward either train or validation loader to update the graph of those
        """
        # Load anchor, positive and negative
        anchor = batch['anchor']
        # Forward method
        h_anchor = self.forward(anchor)

        return {'h_anchor': h_anchor, 'anchor_id': batch['anchor_id'], 'gt_anchor': batch['gt_anchor']}
