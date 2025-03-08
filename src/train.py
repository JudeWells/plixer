import os
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import math

from src.utils import rich_utils

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
# This must be done at the beginning of the program
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    batch_size = cfg.data.config.batch_size
    target_samples_per_batch = cfg.data.config.get("target_samples_per_batch", batch_size)

    # Calculate accumulate_grad_batches
    accumulate_grad_batches = max(1, math.ceil(target_samples_per_batch / batch_size))

    # Set accumulate_grad_batches in trainer config
    cfg.trainer.accumulate_grad_batches = accumulate_grad_batches

    log.info(f"Calculated accumulate_grad_batches: {accumulate_grad_batches}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        except Exception as e:
            log.info(f"Error during training: {e}")
            save_dir = cfg.callbacks.model_checkpoint.dirpath
            os.makedirs(save_dir, exist_ok=True)
            ckpt_save_path = os.path.join(
                save_dir,
                "interrupted.ckpt"
            )
            trainer.save_checkpoint(ckpt_save_path)
            log.info(f"Saved checkpoint to {ckpt_save_path}")
            raise e
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == "__main__":
    main()