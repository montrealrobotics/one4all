from typing import Dict, Union, Optional

import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_f1_score


class HingeLoss:
    """
    Hinge Loss

    DO NOT USE THIS ON UNSHUFFLED DATA: keep in mind this uses all other anchors as negatives. If the data is not
    shuffled, the next anchor will likely be a positive (since the data is ordered by timestep). This means we end
    up sampling a number of false negatives.

        Args:
        :param pos_d: Distance to positive sample
        :param neg_d: Distance to negative sample
        :param rot_pos_d: (NOT_USED) Distance to positive sample (for rotation)
        :param speed_target: (NOT_USED) Use ground truth samples to estimate positive and negative target
        :param scale: Scale the positive and negative target
        :param margin: (NOT_USED) Margin between positive and negative target - only used with speed target
        :param loss: Loss type, options are [l1, l2 , huber]
        :param tau: (NOT_USED) Temperature parameter to scale the loss
        :param hinge_loss: Whether to hinge loss or not
    """

    def __init__(self, pos_d: float = 1.0, neg_d: float = 2.0, rot_pos_d: float = 1.0,
                 speed_target: bool = False, scale: Union[int, float] = 1.0,
                 margin: str = 1.4, loss: str = 'huber', tau: float = 1.0, hinge_loss: bool = True):
        self.speed_target = speed_target
        self.neg_target = 0
        self.scale = scale
        self.margin = margin
        self.loss = loss
        self.tau = tau
        self.pos_d = pos_d * self.scale
        self.neg_d = neg_d * self.scale
        self.rot_pos_d = rot_pos_d * self.scale
        self.hinge_loss = hinge_loss
        assert self.loss in {'l1', 'l2', 'huber'}, 'Please select a valid loss from the options {l1, l2, huber}'

        if self.loss == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        elif self.loss == 'huber':
            self.criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, h: torch.Tensor, batch: Dict,
                 h_aug: Union[bool, torch.Tensor] = False) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss for local backbone.

        Args:
            h: Embedding of anchor and positives, respectively.
            batch: Batch dictionary.
            h_aug: Augmented version of anchor for symmetric part of loss. If false this loss is not computed

        Returns:
            Loss output, top1 and top5 accuracy of positives.
        """
        # Main loss computation
        half_batch = h.size(0) // 2
        # Obtain actions - only half batch as others are used for artificial negatives
        target = batch['action'][:half_batch].squeeze().long()
        # Fetch rotations - for
        index_rotations = target > 2

        # Mask out self (diagonal)
        self_mask = torch.eye(h.size(0), dtype=torch.bool, device=h.device)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=h.size(0) // 2, dims=0)
        neg_mask = torch.logical_xor(~pos_mask, self_mask)
        # Grab only first batch_size // 2 rows
        pos_mask = pos_mask[:half_batch, :]
        neg_mask = neg_mask[:half_batch, :]
        # Compute distance between anchors and negatives
        d = torch.cdist(h, h, p=2)[:half_batch, :]
        # d(x_t, x_{t+1}) Similar pairs
        d_pos = torch.masked_select(d, pos_mask)
        # d(x_t, x_{t+1}) Negative pairs
        d_neg = torch.masked_select(d, neg_mask).view(half_batch, h.size(0) - 2)

        # Compute final loss
        pos_target = torch.ones_like(d_pos) * self.pos_d
        # Fill positive target for rotations with different value
        pos_target.masked_fill_(index_rotations, self.pos_d)
        pos_term = self.criterion(d_pos, pos_target)
        if self.hinge_loss:
            neg_mask = torch.clamp(self.neg_d - d_neg, min=0) > 0
            neg_term = self.criterion(d_neg * neg_mask, (self.neg_d * neg_mask).double()).mean(-1)
        else:
            neg_term = self.criterion(d_neg, torch.ones_like(d_neg) * self.neg_d).mean(-1)

        # Compute symmetric term if h_aug is provided
        sym_term = 0
        if h_aug is not False:
            sym_d = torch.norm(h[:half_batch] - h_aug, dim=-1)
            sym_term = self.criterion(sym_d, torch.zeros_like(sym_d))

        # Compute final loss
        loss = (sym_term + pos_term + neg_term).mean()

        # Get ranking position of positive example
        comb_d = torch.cat(
            [d_pos[:, None], d_neg],  # First position positive example
            dim=-1
        )
        d_argsort = comb_d.argsort(dim=-1, descending=False).argmin(dim=-1)
        # Top1 and Top5 accuracy of positive detection
        acc_top1 = (d_argsort == 0).float().mean()
        acc_top5 = (d_argsort < 5).float().mean()

        metrics = dict(loss=loss, neg_term=neg_term.mean(), pos_term=pos_term.mean(), acc_top1=acc_top1,
                       acc_top5=acc_top5)
        if isinstance(sym_term, torch.Tensor):
            metrics.update({'sym_term': sym_term.mean()})

        return metrics


class CEConnectivity:
    """
    Simple cross-entropy loss for classification with first class being connectivity.

    We report overall metrics, as well as separate metrics over connectivity and actions.
    Args:
        contrastive_loss: Contrastive loss used alongside connectivity classifier
        gt_mse: Only used for research, monitoring model. Always set to False otherwise ground truth data will be used.
        alpha: Regularization parameter between contrastive loss and classification loss
    """

    def __init__(self, contrastive_loss: Optional[HingeLoss] = None, gt_mse: bool = False, alpha: float = 1.0):
        # Classifier loss
        self.clf_loss = nn.CrossEntropyLoss()
        # Contrastive Loss
        self.contrastive_loss = contrastive_loss
        self.mse = nn.MSELoss()
        self.gt_mse = gt_mse
        # Weighting param for contrastive loss
        self.alpha = alpha

    def __call__(self, emb: torch.Tensor, batch: Dict, p_anchor_aug: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classification loss + contrastive computation. The method expects the anchors over the first half of the batch 
        of embeddings and the other half as the positives.

        Args:
            emb: State embeddings - useful when only using hinge contrastive loss class. Ignored.
            batch: Batch dictionary.
            p_anchor_aug: Embedding of augmented anchors - useful when only using hinge contrastive loss class. Ignored

        Returns:
            Hinge loss, top1 and top5 accuracy of positives.
        """
        logits = batch['logits']
        target = batch['action'].squeeze().long()
        n_classes = logits.shape[1]

        # Compute criterion for positives and connectivity score
        # This should ensure they always have equal weight in the loss
        is_negative = target == 0
        is_positive = target > 0

        # Negatives log likelihood term
        neg_nll = self.clf_loss(logits[is_negative], target[is_negative]) if is_negative.any() else 0.0
        # Positive log likelihood term
        # In expectation, half of the elements are stops and assume other classes have uniform weights
        # Weights <- 1 / [0, 1/2, 1 / (2 * n_classes - STOP), 1 / (2 * n_classes - STOP), 1 / (2 * n_classes - STOP)]
        pos_weights = torch.zeros_like(logits[0, :])
        # Inverse frequency
        pos_weights[1] = 2
        pos_weights[2:] = logits[0, 2:].size(0) / 0.5
        pos_nll = torch.nn.functional.cross_entropy(logits[is_positive],
                                                    target[is_positive],
                                                    weight=pos_weights,
                                                    reduction='mean') if is_positive.any() else 0.0

        # Add contrastive regularization
        codes = torch.cat((batch['anchor_codes'], batch['positive_codes']), dim=0)
        if self.contrastive_loss is not None:
            cont_loss = self.contrastive_loss(codes, batch, p_anchor_aug)['loss']
        else:
            cont_loss = 0.0

        # Add ground truth regression target
        # WARNING: This is a sanity check, do not use for final models
        if self.gt_mse:
            targets = torch.cat((batch['gt_anchors'], batch['gt_positives']), dim=0)
            output_dim, gt_dim = codes.shape[1], targets.shape[1]
            if output_dim < gt_dim:
                raise ValueError('Output dim should be at least gt dim if using self.gt_mse')
            elif output_dim > gt_dim:
                # Zero padding
                targets_tmp = torch.zeros_like(codes)
                targets_tmp[:, :gt_dim] = targets
                targets = targets_tmp
            mse = self.mse(codes, targets)
        else:
            mse = 0.0

        # Loss
        loss = ((neg_nll + pos_nll) / 2) + self.alpha * cont_loss + mse

        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)

            # Compute connectivity accuracy
            acc_conn = ((preds > 0) == is_positive).float().mean()

            # Compute action accuracy
            if is_positive.any():
                acc_actions = (preds[is_positive] == target[is_positive]).float().mean()
            else:
                # No positive in batch
                # Above yields nans if there are no positives in batch
                # Tends to happen with small batches
                acc_actions = .5

            # Compute multiclass f1
            # Average over all positive class f1s
            f1_ssl = multiclass_f1_score(preds, target, num_classes=n_classes, average=None)
            f1_ssl = (f1_ssl * (pos_weights / pos_weights.sum()))[1:].sum().item()

            # Average of both
            acc = (acc_conn + acc_actions) / 2

        return dict(
            loss=loss,
            pos_nll=pos_nll,
            neg_nll=neg_nll,
            cont_loss=cont_loss,
            acc_all=acc,
            acc_conn=acc_conn,
            acc_actions=acc_actions,
            pred_clf=preds,
            target_clf=target,
            gt_mse=mse,
            f1_ssl=f1_ssl,
        )
