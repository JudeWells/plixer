from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Mirror provenance to the loggers so it survives on the W&B server even if the
    # local run directory is lost (which is what happened to previous crashed runs).
    provenance = object_dict.get("provenance")
    if provenance:
        git = provenance.get("git", {})
        hparams["provenance/run_id"] = provenance.get("run_id")
        hparams["provenance/git_commit"] = git.get("commit")
        hparams["provenance/git_describe"] = git.get("describe")
        hparams["provenance/git_branch"] = git.get("branch")
        hparams["provenance/git_dirty"] = git.get("is_dirty")
        hparams["provenance/run_dir"] = provenance.get("run_dir")
        hparams["provenance/hostname"] = provenance.get("env", {}).get("hostname")
        hparams["provenance/parent_checkpoints"] = [
            {"config_key": p.get("config_key"), "path": p.get("path")}
            for p in provenance.get("parent_checkpoints", [])
        ]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
