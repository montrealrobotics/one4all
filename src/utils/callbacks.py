import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer
import torch
from src.utils import get_logger

log = get_logger(__name__)


class ModelCheckpointWarmup(pl.callbacks.ModelCheckpoint):
    """Checkpoint callback with additional warmup argument.

    Set warmup to -1 to effectively use checkpointing without warmup.

    This callback will not checkpoint the model during warmup."""

    def __init__(self, warmup=10, **kwargs):
        super(ModelCheckpointWarmup, self).__init__(**kwargs)
        self.warmup = warmup

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > self.warmup:
            super(ModelCheckpointWarmup, self).on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch > self.warmup:
            super(ModelCheckpointWarmup, self).on_validation_end(trainer, pl_module)


class FinetuneConnectivityHeadCallback(BaseFinetuning):
    """
    Callback implementation of finetuning model
    """

    def __init__(self, freeze_at_epoch: int = 100):
        super(FinetuneConnectivityHeadCallback, self).__init__()
        self._freeze_at_epoch = freeze_at_epoch

    def freeze_before_training(self, pl_module):
        """
        Needed before starting training
        """
        pass

    def finetune_function(self, pl_module: pl.LightningModule, current_epoch: int,
                          optimizer: Optimizer, optimizer_idx: int):
        """
        Freeze the whole backbone after freeze_at_epoch number of epochs have passed
        """
        best_model_path = pl_module.trainer.checkpoint_callback.best_model_path
        if best_model_path is not '' and current_epoch == self._freeze_at_epoch:
            # Load best model
            best_model_state = torch.load(best_model_path)['state_dict']
            # Update weights of backbone and connectivity head with best checkpoint
            pl_module.load_state_dict(best_model_state)
            # LocalMetric.load_from_checkpoint(self.hparams.local_metric_path)
            log.info("Freezing model backbone, finetuning connectivity head only")
            log.info("Loading best model {}".format(best_model_path))
            self.freeze(pl_module.net, train_bn=True)
            # Clear optimizer and train head only
            # original_lr = pl_module.trainer.optimizers[0].param_groups[0]['lr']
            # pl_module.trainer.optimizers[0].param_groups.clear()
            # pl_module.trainer.optimizers[0].state.clear()
            # pl_module.trainer.optimizers[0].add_param_group(
            #     {'params': [p for p in pl_module.connectivity_head.parameters()], "lr": original_lr / 10.0})


class FinetuneVingPCallback(BaseFinetuning):
    """
    Callback implementation for Finetuning the ViNG P head.
    """

    def __init__(self, freeze_at_epoch: int = 100):
        super(FinetuneVingPCallback, self).__init__()
        self._freeze_at_epoch = freeze_at_epoch

    def freeze_before_training(self, pl_module):
        """
        Needed before starting training
        """
        pass

    def finetune_function(self, pl_module: pl.LightningModule, current_epoch: int,
                          optimizer: Optimizer, optimizer_idx: int):
        """
        Freeze the whole backbone after freeze_at_epoch number of epochs have passed
        """
        best_model_path = pl_module.trainer.checkpoint_callback.best_model_path
        if best_model_path is not '' and current_epoch == self._freeze_at_epoch:
            # Load best model
            best_model_state = torch.load(best_model_path)['state_dict']
            # Update weights of backbone and connectivity head with best checkpoint
            pl_module.load_state_dict(best_model_state)
            # LocalMetric.load_from_checkpoint(self.hparams.local_metric_path)
            log.info("Freezing model backbone and T head, finetuning P head")
            log.info("Loading best model {}".format(best_model_path))
            self.freeze(pl_module.net, train_bn=True)
            self.freeze(pl_module.T, train_bn=True)
            # Clear optimizer and train head only
            # original_lr = pl_module.trainer.optimizers[0].param_groups[0]['lr']
            # pl_module.trainer.optimizers[0].param_groups.clear()
            # pl_module.trainer.optimizers[0].state.clear()
            # pl_module.trainer.optimizers[0].add_param_group(
            #     {'params': [p for p in pl_module.connectivity_head.parameters()], "lr": original_lr / 10.0})
