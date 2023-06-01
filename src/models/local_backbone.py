from typing import Dict, List, Optional, Tuple, Union
import functools

from tqdm import tqdm
import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange
import pytorch_lightning as pl

from src.models.base_model import BaseModel
from src.utils.visualization import plot_2d_latent, plot_connectivity_graph
from src.utils.visualization import visualize_spurious_edges, plot_spurious_edges_graph
from src.utils.eval_metrics import GraphMetrics, edges_minus_chains_recovered, aggregate_val_metrics
from src.models.loss import HingeLoss, CEConnectivity
from src.utils import get_logger

log = get_logger(__name__)


class LocalBackbone(BaseModel):
    """
    Local backbone network implementation for a Maze/Habitat/JAckal Environment.

        Args:
            net: Local Backbone for Maze or Jackal.
            connectivity_head: Net used for predicting connectivity + actions.
            loss: Loss function, options are CEConnectivity or HingedLoss
            optimizer: functools partial torch.optim optimizer with all arguments except
            model parameters. Please refer to the Hydra doc for partially initialized modules.
            n_samples: Number of false positives edges used to plot
            rho: Min distance to consider an edge spurious
            n_edges: Number of edges to sample while plotting connectivity graph. Note: Only used for
            monitoring/research - not used by the actual O4A model
            log_figure_every_n_epoch: Frequency at which figures should be logged. The counter only increases when
            validation_epoch_end is called
            val_loss: Validation Loss function, options are CEConnectivity or HingedLoss. Is False, use same as loss.
            lr_scheduler_config: lr_scheduler_config dict. Please refer to pytorch lightning doc.
            The scheduler arg should be a functools.partial missing only the optimizer
            step_every: Number of environment batches to forward before calling an optimizer step. Useful when using
            per_env_batches and a "batch" consists of a large number of environment batches
    """

    def __init__(self,
                 net: pl.LightningModule,
                 connectivity_head: pl.LightningModule,
                 loss: Union[CEConnectivity, HingeLoss],
                 optimizer: functools.partial,
                 n_samples: int,
                 rho: float,
                 n_edges: int,
                 log_figure_every_n_epoch: int = 1,
                 val_loss: Union[CEConnectivity, HingeLoss, None] = None,
                 lr_scheduler_config: Dict = None,
                 step_every: int = 4):
        super(LocalBackbone, self).__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Classification metrics to measure quality of connectivity 
        # Note: Only used for monitoring/research - not used by the actual O4A model
        self.metrics = GraphMetrics()

        self.loss = loss
        self.val_loss = val_loss if val_loss is not None else self.loss  # Use same loss as training
        # Map model to double
        self.net = net
        self.connectivity_head = connectivity_head
        # Used to use all other examples in batch as negatives for a given anchor
        self.contrast = True
        self.has_logged_debug_gt = True  # Set false to log ground truth graph once

        # Manual backwards
        self.automatic_optimization = False

    def log_figures(self) -> bool:
        return super().log_figures() or not self.has_logged_debug_gt

    def forward(self, x_anchor: torch.Tensor, x_pos: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                                                   torch.Tensor,
                                                                                   torch.Tensor]:
        """
        Local backbone forward method

        Args:
            x_anchor: Batch of sequence of images (anchors) [B, S, C, H, W]
            x_pos: Batch of positives for anchors [B, S, C, H, W]

        Returns:
            Embedding representation of sequence of anchors [B, D]
            Embedding representation of positives [B, D]
            Embedding representation of each anchor in the sequence (backbone) [B, K, D]
        """
        # Forward through Local backbone network
        p_anchor, p_pos = self.net.forward(x_anchor, x_pos)

        if p_pos is not None:
            if self.contrast:
                # Always do this during training. Turn OFF during deployment. Expand batch to get more negatives
                # Compare anchors to anchors and we will treat them as negatives or STOP later
                p_anchor_1 = p_anchor.repeat_interleave(p_anchor.shape[0], dim=0)
                p_anchor_2 = p_anchor.repeat((p_anchor.shape[0], 1))

                p_anchor_input = torch.cat((p_anchor, p_anchor_1), dim=0)
                p_pos_input = torch.cat((p_pos, p_anchor_2), dim=0)
            else:
                p_anchor_input = p_anchor
                p_pos_input = p_pos
            logits = self.connectivity_head(p_anchor_input, p_pos_input)
            p_pos = p_pos.double()
        else:
            p_pos = None
            logits = None

        return p_anchor.double(), p_pos, logits

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
            metrics, p_anchor = self._shared_loss_computation(b, self.loss)

            if 'pred_clf' in metrics.keys():
                metrics.pop('pred_clf')
                metrics.pop('target_clf')

            # Log metrics
            log_metrics = {'train/' + key: val for key, val in metrics.items()}
            self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Backward the loss iteratively to save GPU space
            # This is implicitly like summing all the loss terms
            loss = metrics['loss'] / len(batch)
            self.manual_backward(metrics['loss'] / len(batch))  # Multiply the average factor for each env
            loss_all += loss.item()

            if (i + 1) % self.hparams.step_every == 0.0 or i == n_batches - 1:
                opt.step()
                opt.zero_grad()

        # Only for reporting
        self.log_dict({"train/loss_avg": loss_all}, prog_bar=True, logger=True)

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int,
                        dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Validation loop
        """
        # Compute loss and anchor embedding
        metrics, p_anchor = self._shared_loss_computation(val_batch, self.val_loss)
        pred_clf = metrics.pop('pred_clf')
        target_clf = metrics.pop('target_clf')

        # Condition to fetch environment properly if number of environments is equal to one
        dataloader_idx = 0 if dataloader_idx is None else dataloader_idx
        # Fetch environment name
        env = self.fetch_environment(stage='val', dataloader_idx=dataloader_idx)

        # Log metrics
        log_metrics = {'val/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        return {'pred': p_anchor,
                'target': val_batch['gt_anchors'],
                'pred_id': val_batch['anchors_id'],
                'dataloader_idx': dataloader_idx, 'stage': 'val', 'pred_clf': pred_clf, 'target_clf': target_clf}

    def test_step(self, test_batch: Dict[str, torch.Tensor], batch_idx: int,
                  dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Test loop
        """
        # Compute loss and anchor embedding
        metrics, p_anchor = self._shared_loss_computation(test_batch, self.val_loss)
        metrics.pop('pred_clf')
        metrics.pop('target_clf')

        # Condition to fetch environment properly if number of environments is equal to one
        dataloader_idx = 0 if dataloader_idx is None else dataloader_idx
        env, prefix = self.get_test_dataloader_env(dataloader_idx)

        # Log metrics
        log_metrics = {prefix + '/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        return {'pred': p_anchor,
                'target': test_batch['gt_anchors'],
                'pred_id': test_batch['anchors_id'],
                'dataloader_idx': dataloader_idx, 'stage': prefix}

    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Log validation loss
        log_metrics = aggregate_val_metrics(self.trainer.logged_metrics)
        self.log_dict(log_metrics, prog_bar=True, logger=True)

        """Report various metrics over the whole validation split."""
        if self.log_figures():
            self._shared_validation_epoch_end(validation_step_outputs, self.trainer.val_dataloaders)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Log averaged validation loss
        log_metrics = aggregate_val_metrics(self.trainer.logged_metrics)
        self.log_dict(log_metrics, prog_bar=True, logger=True)
        # To enable test figures logging
        self.figure_counter = 0
        self.hparams.log_figure_every_n_epoch = 1
        self._shared_validation_epoch_end(outputs, self.trainer.test_dataloaders)

    def _shared_validation_epoch_end(self,
                                     outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]],
                                     dataloaders) -> None:
        """
        Test epoch end loop.
        """
        if len(dataloaders) == 1:
            # Outputs is a list of dicts because of a single data loader
            # Wrap in list for compatibility with multiple dataloader case
            outputs = [outputs]

        # Iterate over each env output if using multi-environments
        class_metrics = dict()

        for data in outputs:
            # Multiple environments
            # Just pick first element in the list as they all are the same
            dataloader_idx = data[0]["dataloader_idx"]

            # Fetch environment
            dataloader = dataloaders[dataloader_idx]
            env = dataloader.dataset.experiment
            stage = data[0]["stage"]

            # If number of environments is equal to one, make list of dicts
            class_metrics.update(self._validation_metrics(data, env, dataloader, stage))

            if "pred_clf" in data[0].keys() and self.logger is not None:
                # Log confusion matrix
                pred_clf = torch.cat([x["pred_clf"] for x in data]).detach().cpu().numpy()
                target_clf = torch.cat([x["target_clf"] for x in data]).detach().cpu().numpy()
                comet_ml = self.logger.experiment
                if hasattr(comet_ml, 'log_image'):
                    comet_ml.log_confusion_matrix(target_clf, pred_clf, title=env)

        # Aggregate classification metrics
        log_class_metrics = aggregate_val_metrics(class_metrics)
        self.log_dict(log_class_metrics, prog_bar=True, logger=True)

    def _shared_loss_computation(self, batch: Dict[str, torch.Tensor], loss: HingeLoss) \
            -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute loss for the local backbone network

        Args:
            batch: Either train or validation batch

        Returns:
            Loss value for given batch and anchors embeddings
        """
        # Extract datamodule
        anchor, pos = batch['anchors'], batch['positives']
        # Forward method
        p_anchor, p_pos, logits = self.forward(x_anchor=anchor, x_pos=pos)
        batch['anchor_codes'] = p_anchor
        batch['positive_codes'] = p_pos
        batch['logits'] = logits

        # Massage target
        target = batch['action']

        # Forward uses all other anchors as negatives during training
        batch_size = anchor.shape[0]
        n_samples = logits.shape[0]

        if self.contrast:
            neg_target = torch.zeros((n_samples - batch_size,), device=target.device).long()
            neg_target[::batch_size + 1] = 1  # Those are anchor-anchor exact comparisons and are STOP
            batch['action'] = torch.cat((target.squeeze(), neg_target), dim=0)

        # Fetch augmented version of anchors if available and forward
        p_anchor_aug = self.forward(x_anchor=batch['anchors_aug'])[0] if batch.get('anchors_aug') is not None else False
        if p_anchor_aug is not False:
            aug_anchor_logits = self.connectivity_head(p_anchor.float(), p_anchor_aug.float()).double()
            # Create logits target for zero distance target at the end of tensor
            aug_anchor_target = torch.ones((aug_anchor_logits.size(0),), device=target.device).long()
            batch['action'] = torch.cat((batch['action'], aug_anchor_target), dim=0)
            # Populate logits
            batch['logits'] = torch.cat((batch['logits'], aug_anchor_logits), dim=0)
        h = torch.cat((p_anchor, p_pos), dim=0)
        # Compute loss
        metrics = loss(h, batch, p_anchor_aug)
        # Compute norm between anchors and positives
        norm = torch.linalg.norm(p_anchor - p_pos, dim=1).mean()
        metrics['norm'] = norm

        return metrics, p_anchor

    def _validation_metrics(self, validation_step_outputs: List[Dict[str, torch.Tensor]], env: str, dataloader,
                            stage: str = 'val') -> Dict[str, float]:
        """
        Compute classification metrics over graph, compute 2 first components of anchor embeddings and loop-closure of
        graph.

        Args:
            validation_step_outputs: Validation output of an environment for a given epoch
            env: String describing the env.
            dataloader: Dataloader of current data.
            stage: Current stage, options are validation or testing

        Returns:
            Classification metrics for given environment
        """
        # Extract predictions, ids and gt
        pred = torch.cat([x["pred"] for x in validation_step_outputs]).detach().cpu().numpy()
        pred_id = torch.cat([x["pred_id"] for x in validation_step_outputs]).detach().cpu().numpy()
        target = torch.cat([x["target"] for x in validation_step_outputs]).detach().cpu().numpy()
        # Fetch ground truth graph
        gt_graph = dataloader.dataset.get_ground_truth_graph()
        chain_graph = dataloader.dataset.chain_graph

        # Compute and plot connectivity of the graph given current local backbone
        pos_conn_fig, rot_conn_fig, graph = self._connectivity_graph(dataloader, pred,
                                                                     pred_id, gt_graph.number_of_edges())
        # Plot bad connections (fig) - If fig is None it means there are not spurious edges
        bad_conn_fig, spurious_edges, worst_edge_dist = visualize_spurious_edges(gt_graph, graph,
                                                                                 self.hparams.rho, dataloader.dataset,
                                                                                 self.hparams.n_samples)

        # Plot bad connections (graph) - If fig is None it means there are not spurious edges
        bad_conn_graph_fig = None
        if bad_conn_fig is not None:
            bad_conn_graph_fig = plot_spurious_edges_graph(graph, spurious_edges=spurious_edges,
                                                           environment=dataloader.dataset.env_type)

        # Classification metrics recovering non-transition edges
        metrics = edges_minus_chains_recovered(gt_graph, chain_graph, graph)
        log_metrics = {stage + '/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True)

        # Compute classification metrics for graph connectivity
        # Note: Only used for monitoring/research - not used by the actual O4A model
        metrics = self.metrics(graph, gt_graph)
        metrics['worst_edge_d'] = worst_edge_dist
        log_metrics = {stage + '/' + key + '/' + env: val for key, val in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True)

        # Plot local backbone embeddings
        emb_fig = plot_2d_latent(pred, target[:, 0] + target[:, 1])

        if self.logger is not None:
            current_epoch = self.trainer.current_epoch
            # Log embeddings, position graph, rotation graph, images of spurious edges and spurious edges's graph
            self.log_fig2image(emb_fig, f'{stage}_{env}_local_embedding_epoch_{current_epoch}')
            self.log_fig2image(pos_conn_fig, f'{stage}_{env}_local_pos_connectivity_epoch_{current_epoch}')
            self.log_fig2image(rot_conn_fig, f'{stage}_{env}_local_rot_connectivity_epoch_{current_epoch}')
            self.log_fig2image(bad_conn_fig, f'{stage}_{env}_bad_connections_epoch_{current_epoch}')
            self.log_fig2image(bad_conn_graph_fig, f'{stage}_{env}_bad_connections_graph_epoch_{current_epoch}')

        return log_metrics

    def _connectivity_graph(self, dataloader: DataLoader,
                            anchors: np.ndarray,
                            ids: np.ndarray,
                            number_gt_edges: int) -> object:
        """
        Update the graph edges based on current local backbone estimate and plot the graph.
        Note: Method used for monitoring performance of the local backbone

        Args:
            dataloader: Dataloader used to update graph
            anchors: Embedding for each node in graph
            ids: Ids of each anchor in graph
            number_gt_edges: Actual or approximate number of GT edges to prevent memory from exploiting. Note: Only 
            used for monitoring/research - not used by the actual O4A model

        Returns:
            NetworkX updated graph with new edges
        """
        # Update training graph - do not update graph attribute in validation dataloader
        if self.has_logged_debug_gt:
            graph = self.update_graph(anchors=anchors, indices=ids, batch_size=32, update=False,
                                      dataset=dataloader.dataset, number_gt_edges=number_gt_edges)
        else:
            # Log debug ground truth graph
            graph = dataloader.dataset.get_ground_truth_graph()
            self.has_logged_debug_gt = True

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

        return figures[0], figures[1], graph

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int,
                     dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Prediction step, forward either train or validation loader to update the graph of those
        """
        # Load anchor, positive and negative
        anchor = batch['anchors']
        # Turn contrast off
        self.contrast = False

        # Forward method but only using anchors
        if hasattr(self, 'predict_pos') and self.predict_pos:
            x_pos = batch['positives']
        else:
            x_pos = None

        # Forward anchor and positive
        p, pos, logits = self.forward(x_anchor=anchor, x_pos=x_pos)

        # Add extra dimension as predict step turns drop_last to false automatically
        batch['anchors'] = p if p.ndim > 1 else p[None, :]

        if hasattr(self, 'predict_pos') and self.predict_pos:
            batch['positives'] = pos if pos.ndim > 1 else pos[None, :]
        else:
            # Don't keep positives in batch otherwise we run OOM on large datasets
            keys = list(batch.keys())
            for key in keys:
                if "positives" in key:
                    batch.pop(key)
        return batch

    def update_graph(self, anchors: np.ndarray, indices: np.ndarray,
                     batch_size: int, dataset: Dataset,
                     update: bool = False,
                     number_gt_edges: int = 1e6) -> nx.Graph:
        """
        Update graph using local backbone and connectivity head.

        Args:
            anchors: Training predictions for the anchors
            indices: State id of the predicted anchors
            batch_size: Original batch size of the model
            dataset: Used to retrieve useful information like the graph
            update: Whether if updated graph attribute in class should be updated with updated graph or not
            number_gt_edges: Number of GT edges

        Returns:
            Updated graph
        """

        counter = dict(closures=0, transitions=0, all=0)
        log.info('Updating {} graph for {}...'.format(dataset.split_raw, dataset.experiment))

        if torch.is_tensor(indices):
            # Remap to list as torch tensors generate issues with networkx
            indices = indices.view(-1).detach().tolist()

        # Create temporal graph object
        graph = nx.create_empty_copy(dataset.graph)
        # Placeholder with list of edges
        edges = []

        # Computation of action probabilities
        anchors = torch.from_numpy(anchors).to(self.connectivity_head.device)

        for index, anchor in tqdm(zip(indices, anchors), total=len(indices), desc='Updating graph...'):
            # If we are accumulating too much edges, break the cycle
            # Note: This number does not have to be the ground truth number of edges but any arbitrarily large number
            if len(edges) > number_gt_edges * 15:
                log.info(f"Breaking, there are 15 times more edges than in GT for {dataset.experiment}")
                log.info(f"Number of GT edges {number_gt_edges} - Number of predicted edges {len(edges)}")
                break
            # Derive connectivity using connectivity head
            with torch.inference_mode():
                # Compute local distance
                weights = rearrange(torch.cdist(rearrange(anchor, 'd -> 1 d'), anchors), '1 b -> b')

                # Repeat anchor n-samples times
                anchor = torch.repeat_interleave(rearrange(anchor, 'd -> 1 d'), repeats=anchors.shape[0], dim=0)
                # Compute locomotion predictions on each (anchor, positive) pair
                # This is for a given anchor, compute the prob score against all other anchors
                logits = self.connectivity_head(anchor.float(), anchors.float())
                actions = logits.argmax(dim=1)

            # Use this to get geodesic for forward move, rotation, etc.
            weights = weights.cpu().numpy()
            actions = actions.cpu().numpy()
            counter['closures'] += (actions > 0).sum()

            if update:
                # If updating graph, add back transition edges from the original graph of chains
                # We only do so if the model predicted not connected for that edge
                for i_r, r in enumerate(index):
                    for i_c, c in enumerate(indices):
                        if dataset.graph.has_edge(c, r):
                            # Replace predicted action in graph with GT action
                            # TODO: this will break as if edge (i,j) has action, edge (j,i) will not have action
                            actions[i_r, i_c] = dataset.graph.edges[c, r]['action'][0]
                            if actions[i_r, i_c] == 0:
                                # Use graph here, it should not be updated, and it is directed whereas chain_graph is undirected
                                counter["transitions"] += 1


            # Extract selected edges + weight and add to list of edges
            if actions.any():
                columns = np.nonzero(actions)[0]  # 0 is disconnected
                weights = weights[columns]
                actions = actions[columns]
                batch_edges = [(index, indices[c], {'weight': w, 'action': a}) for c, w, a in zip(columns,
                                                                                                  weights,
                                                                                                  actions)]
                edges.extend(batch_edges)

        log.info('Edges successfully computed')
        log.info(f'Transition edges   : {counter["transitions"]}')
        log.info(f'Loop closure edges : {counter["closures"]}')
        log.info(f'Total edges        : {counter["transitions"] + counter["closures"]}')
        # Populate graph with current edges list
        graph.add_edges_from(edges)
        # Update graph used to sample neighbors in class with updated one or keep using original one
        dataset.graph = graph.copy() if update else dataset.graph.copy()
        log.info('Graph successfully computed, environment {}'.format(dataset.experiment))
        return graph

    def on_after_backward(self):
        """Adapted from https://github.com/Lightning-AI/lightning/issues/5238."""
        if self.trainer.global_step % 5 == 0:
            avg_abs_grad = dict(
                grad_encoder=self.avg_abs_grad(self.net.encoder),
                grad_predictor=self.avg_abs_grad(self.net.predictor),
                grad_connectivity=self.avg_abs_grad(self.connectivity_head),
            )
            log_metrics = {'train/' + key + '/all_env': val for key, val in avg_abs_grad.items()}
            self.log_dict(log_metrics, prog_bar=False, logger=True)
