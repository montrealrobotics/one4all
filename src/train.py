import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import comet_ml

from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.models.local_metric import LocalMetric
from src import utils

log = utils.get_logger(__name__)


def train(cfg: DictConfig) -> Optional[float]:
    """
    Main training loop for all components.

    Please refer to the class docstring or the yaml configs for model and datamodule arguments.

    Args:
        cfg: Config with all hyperparameters for training
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        utils.seed_experiment(cfg.seed)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.setup_components()  # Initialize backbones and heads

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger_ = hydra.utils.instantiate(lg_conf)
                # Add comet tag if using comet logger
                if cfg.logger.get('comet'):
                    logger_.experiment.add_tag(cfg.logger.tag)
                    logger_.experiment.log_asset_data(OmegaConf.to_container(cfg), 'config.yaml')
                logger.append(logger_)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Fitting model")
    trainer.fit(model=model, datamodule=datamodule)

    # If we have a logger, log the checkpoint
    if cfg.get('logger'):
        # If we are using comet loger, log using specialized methods
        if cfg.logger.get('comet'):
            name = cfg.name + '_' + cfg.environment  # Model name for Comet
            best_model_path = trainer.checkpoint_callback.best_model_path
            best_model_name = os.path.basename(best_model_path)
            best_epoch = int(''.join(filter(str.isdigit, best_model_name)))
            trainer.loggers[0].experiment.log_model(file_or_folder=trainer.checkpoint_callback.best_model_path,
                                                    name=name, file_name=name + '_' + best_model_name)
            # Log best model and best validation loss
            trainer.loggers[0].experiment.log_metric('best_val_loss',
                                                     trainer.checkpoint_callback.best_model_score,
                                                     epoch=best_epoch)

    # Test the model
    trainer.test(dataloaders=datamodule.test_dataloader())
    trainer.test(dataloaders=datamodule.val_dataloader())

    # Train dataloader may be a ConcatDataset
    datamodule.hparams.per_env_batches = True  # This will force dataloader to be a list of dataloaders
    trainer.test(dataloaders=datamodule.train_dataloader(test_on_train=True))

    # Obtain best score of model
    score = trainer.checkpoint_callback.best_model_score

    return score.item()