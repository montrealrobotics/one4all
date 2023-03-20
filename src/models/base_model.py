"""BaseModel class to add to the different PL modules for common methods"""
from typing import Dict
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from src.utils import fig2img


class BaseModel(pl.LightningModule):
    def __init__(self, freeze_backbone=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.figure_counter = 0
        self.freeze_backbone = freeze_backbone

    def configure_optimizers(self) -> Dict:
        # optimizer is a functools.partial
        # complete init with parameters
        optimizer = self.hparams.optimizer(self.parameters())
        result = dict(optimizer=optimizer)

        if self.hparams.lr_scheduler_config is not None and self.hparams.lr_scheduler_config.scheduler is not None:
            # scheduler is a functools.partial
            # complete init with optimizer
            self.hparams.lr_scheduler_config.scheduler = self.hparams.lr_scheduler_config.scheduler(optimizer)

            # cast scheduler config to dict since lightning relies on this for checks
            result.update(lr_scheduler=OmegaConf.to_container(self.hparams.lr_scheduler_config))

        return result

    def log_figures(self) -> bool:
        """Call to check if we should log figures.

        This can be used to decouple check_val_every_n_epoch and figure logging. Some figures
        (e.g. connectivity) are expensive. We may want to generate them at lower frequency
        than validation, which is useful for checkpointing and scheduling."""
        self.figure_counter += 1
        if self.figure_counter == self.hparams.log_figure_every_n_epoch:
            self.figure_counter = 0
            return True
        return False

    def setup_components(self) -> None:
        """Set up backbone and geodesic regressor."""

        # Load actual backbone for training
        if 'local_metric_path' in self.hparams:
            from src.models.local_metric import LocalMetric
            self.backbone = LocalMetric.load_from_checkpoint(self.hparams.local_metric_path)
            if self.freeze_backbone:
                self.backbone.freeze()

        # Optional global head for better visualizations
        if 'global_metric_path' in self.hparams and self.hparams.global_metric_path is not None:
            from src.models.global_metric import GlobalMetric
            self.global_head = GlobalMetric.load_from_checkpoint(self.hparams.global_metric_path, strict=False)
            self.global_head.freeze()

    def fetch_environment(self, stage: str, dataloader_idx: int) -> str:
        """
        Fetch the environment's name given current training loop

        Args:
            stage: Current stage [train, val, test]
            dataloader_idx (int): Index of current batch

        Returns:
            Name of environment for given batch
        """
        # Obtain dataloader
        if stage == 'train':
            dataloader = self.trainer.train_dataloader
        elif stage == 'val':
            dataloader = self.trainer.val_dataloaders
        elif stage == 'test':
            dataloader = self.trainer.test_dataloaders
        else:
            raise ValueError('stage should be train, val or test')
        # Fetch environment name for given dataloader
        env = dataloader[dataloader_idx].dataset.experiment

        return env

    def get_test_dataloader_env(self, dataloader_idx):
        """More flexible method to fetch test dataloaders.

        This can be called even if we are running test routines on val
        or train dataloaders."""
        # Fetch environment name
        dataloader = self.trainer.test_dataloaders[dataloader_idx]
        env = dataloader.dataset.experiment
        split = dataloader.dataset.split_raw  # Check what current split we have

        # Add _final suffix to val and train
        prefix = split if split == 'test' else split + '_final'

        return env, prefix

    def avg_abs_grad(self, module):
        sum_ = 0.0
        n = 0
        for p in module.parameters():
            grad = p.grad.data.view(-1)
            sum_ += grad.abs().sum()
            n += grad.shape[0]

        return (sum_ / n).item() if n else 0.0

    def log_fig2image(self, figure: object, title: str) -> None:
        """
        Log a figure as an image using logger and close figure.

        Args:
            figure: Matplotlib figure object
            title: Title of the image

        Returns:
            None
        """
        # Get logger
        comet_ml = self.logger.experiment
        if figure is not None:
            comet_ml.log_image(image_data=fig2img(figure), name=title)
            # Close figure
            plt.close(figure)
