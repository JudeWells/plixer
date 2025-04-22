import argparse
import os

import optuna
import rootutils
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from src.train import train as run_train_fn
from src.utils import get_metric_value
from hydra.core.hydra_config import HydraConfig

# Ensure project root is on PYTHONPATH via rootutils (expects .project-root marker)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def hydra_train(overrides):
    """Helper that composes a Hydra config and executes the shared `train` function.

    Args:
        overrides (list[str]): List of override strings passed to Hydra.

    Returns:
        float: The value of the optimized metric (lower is better for loss).
    """
    # We want to treat the repo root as CWD for Hydra so that relative paths match
    repo_root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    os.chdir(repo_root)

    abs_cfg_dir = os.path.join(repo_root, "configs")
    print(f"Set config dir to {abs_cfg_dir}")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    # The shared train() util returns (metric_dict, object_dict)
    metric_dict, _ = run_train_fn(cfg)

    # Extract the metric to be optimised (same key as in cfg.optimized_metric)
    metric_value = get_metric_value(metric_dict, cfg.get("optimized_metric"))
    return metric_value


def build_objective(experiment_name: str, max_epochs: int):
    """Create an Optuna objective function that tunes Poc2Mol hyper‑parameters."""

    def objective(trial: optuna.trial.Trial):
        # Suggest hyper‑parameters
        lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)
        dropout = trial.suggest_float("dropout_prob", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        f_maps = trial.suggest_categorical("fmaps", [64, 128])
        num_levels = trial.suggest_categorical("num_levels", [5, 7, 9, 11])


        # Build Hydra overrides for this trial
        overrides = [
            f"experiment={experiment_name}",
            "ckpt_path=null",  # Always start from scratch during tuning
            f"model.lr={lr}",
            f"model.config.dropout_prob={dropout}",
            f"model.config.f_maps={f_maps}",
            f"model.config.num_levels={num_levels}",
            f"data.config.batch_size={batch_size}",
            f"trainer.max_epochs={max_epochs}",
            "trainer.check_val_every_n_epoch=4",
            # Avoid extensive checkpointing during HPO
            "callbacks.model_checkpoint.save_top_k=0",
            "logger.wandb.offline=True",  # Do not spam W&B during HPO
        ]
        # Execute training for this trial and return the metric
        metric_value = hydra_train(overrides)
        return metric_value

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna sweep for Poc2Mol (HiQBind)")
    parser.add_argument("--experiment", default="train_poc2mol_hiqbind", help="Name of experiment config under configs/experiment")
    parser.add_argument("--study_name", default="poc2mol_hiqbind_optuna", help="Optuna study identifier")
    parser.add_argument("--storage", default="sqlite:///poc2mol_hiqbind_optuna.db", help="Optuna storage URI (SQLite or RDB)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run in this process. Use -1 for infinite loop (useful for parallel workers)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs per trial. Keep small for faster sweeps")
    parser.add_argument("--timeout", type=int, default=None, help="Global timeout (seconds) for the entire optimisation run")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create / connect to the Optuna study (enable distributed optim).
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )

    objective = build_objective(args.experiment, args.max_epochs)

    # The `optuna.study.optimize` call is blocking.  Setting n_jobs=1 because each
    # SGE worker already occupies one GPU. If you want CPU‑only parallelism you
    # can increase this.
    n_trials = None if args.n_trials == -1 else args.n_trials
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
    )

    best = study.best_trial
    print("\n===== Optuna optimisation finished =====")
    print(f"Best trial # {best.number}")
    print(f"  Value (val/loss): {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main() 