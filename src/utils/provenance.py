"""Run provenance: record exactly what code, config and parent checkpoints produced a run.

Motivation
----------
Evaluation directories in this project have historically stored results without any
record of which checkpoint or config produced them, which makes it impossible to say
afterwards which model a given table came from. This module makes every run and every
saved checkpoint self-describing.

Two artefacts are produced:

``<run_dir>/provenance.json`` (plus ``uncommitted.patch`` / ``resolved_config.yaml``)
    Written once at the start of the run.

``checkpoint["provenance"]``
    Embedded in *every* checkpoint the run saves, so a ``.ckpt`` file on its own is
    enough to identify its origin. Because the record includes the parent checkpoints
    the run was seeded from, lineage can be walked backwards - see
    :func:`describe_checkpoint`.

Nothing here is allowed to interrupt training: every collector degrades to an ``error``
entry rather than raising.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from lightning import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

PROVENANCE_KEY = "provenance"
PROVENANCE_FILENAME = "provenance.json"
PATCH_FILENAME = "uncommitted.patch"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

# Packages whose versions materially change results if they drift.
_TRACKED_PACKAGES = (
    "torch",
    "lightning",
    "transformers",
    "rdkit",
    "docktgrid",
    "numpy",
    "setuptools",
)


def _repo_root() -> str:
    """Directory of the repository, derived from this file's location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd or _repo_root(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Only trailing newlines are stripped: `git status --porcelain` encodes the
    # staged/unstaged state in the first two columns, so leading spaces are data.
    return out.stdout.rstrip("\n")


def collect_git_info() -> Dict[str, Any]:
    """Commit, branch, tag-relative description and working-tree cleanliness."""
    info: Dict[str, Any] = {"repo_root": _repo_root()}

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        info["error"] = "not a git repository, or git unavailable"
        return info

    info["commit"] = commit
    info["commit_short"] = commit[:7]
    info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    # Position relative to the nearest tag, e.g. "baseline/paper-v1-14-gab12cd3-dirty".
    info["describe"] = _git("describe", "--tags", "--always", "--dirty")
    info["commit_time"] = _git("show", "-s", "--format=%cI", "HEAD")
    info["commit_subject"] = _git("show", "-s", "--format=%s", "HEAD")
    info["remote"] = _git("config", "--get", "remote.origin.url")

    # Uncommitted changes to tracked files. Untracked files are listed separately
    # because `git diff HEAD` does not capture them, so the saved patch cannot
    # reconstruct them.
    status = _git("status", "--porcelain")
    info["is_dirty"] = bool(status)
    if status:
        modified, untracked = [], []
        for line in status.splitlines():
            code, path = line[:2], line[3:]
            (untracked if code == "??" else modified).append(path)
        info["modified_files"] = modified
        info["untracked_files"] = untracked

    return info


def collect_env_info() -> Dict[str, Any]:
    """Interpreter, host and versions of the packages that affect results."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    try:
        info["user"] = getpass.getuser()
    except Exception:
        pass

    packages: Dict[str, str] = {}
    try:
        import importlib.metadata as importlib_metadata

        for name in _TRACKED_PACKAGES:
            try:
                packages[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue
    except Exception as exc:  # pragma: no cover - defensive
        packages["error"] = str(exc)
    info["packages"] = packages

    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive
        info["cuda"] = {"error": str(exc)}

    # Which GPUs this process was actually allowed to see.
    for var in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "WANDB_MODE"):
        if var in os.environ:
            info.setdefault("env_vars", {})[var] = os.environ[var]

    return info


def _describe_ckpt_file(path: str) -> Dict[str, Any]:
    """Identify a checkpoint file, reading its own provenance record if it has one."""
    record: Dict[str, Any] = {"path": path, "abspath": os.path.abspath(path)}
    if not os.path.exists(path):
        record["exists"] = False
        return record

    record["exists"] = True
    stat = os.stat(path)
    record["size_bytes"] = stat.st_size
    record["mtime"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()

    # If the parent was itself produced under this system it carries its own record,
    # which is what makes lineage walkable. Only the small metadata keys are read.
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        record["epoch"] = ckpt.get("epoch")
        record["global_step"] = ckpt.get("global_step")
        parent = ckpt.get(PROVENANCE_KEY)
        if isinstance(parent, dict):
            record["provenance"] = {
                "run_id": parent.get("run_id"),
                "git_commit": parent.get("git", {}).get("commit"),
                "git_describe": parent.get("git", {}).get("describe"),
                "task_name": parent.get("task_name"),
                "timestamp": parent.get("timestamp"),
                "parent_checkpoints": parent.get("parent_checkpoints"),
            }
        del ckpt
    except Exception as exc:
        record["read_error"] = str(exc)

    return record


def find_parent_checkpoints(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Locate every ``ckpt_path`` in the config and identify the file it points at.

    Checkpoint paths are not only at the top level: the combined experiment embeds a
    Poc2Mol checkpoint inside the train and val dataset configs, and those links are
    precisely what identifies which voxel model generated the decoder's inputs.
    """
    found: List[Dict[str, Any]] = []
    # The combined config points at the same Poc2Mol checkpoint from several places;
    # inspecting a multi-GB file once per reference would be needlessly slow.
    cache: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, DictConfig) or isinstance(node, dict):
            for key in list(node.keys()):
                try:
                    value = node[key]
                except Exception:
                    continue  # unresolvable interpolation
                child_path = f"{path}.{key}" if path else str(key)
                if key == "ckpt_path" and isinstance(value, str) and value:
                    if value not in cache:
                        cache[value] = _describe_ckpt_file(value)
                    entry = dict(cache[value])
                    entry["config_key"] = child_path
                    found.append(entry)
                else:
                    walk(value, child_path)
        elif isinstance(node, (list, tuple)) or type(node).__name__ == "ListConfig":
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    try:
        walk(cfg, "")
    except Exception as exc:  # pragma: no cover - defensive
        found.append({"error": str(exc)})
    return found


def collect_provenance(cfg: DictConfig, run_dir: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the full provenance record for a run."""
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
    }
    try:
        record["task_name"] = cfg.get("task_name")
        record["tags"] = (
            OmegaConf.to_container(cfg.get("tags"), resolve=True) if cfg.get("tags") else None
        )
        record["seed"] = cfg.get("seed")
        record["run_dir"] = run_dir or cfg.get("paths", {}).get("output_dir")
    except Exception as exc:
        record["config_error"] = str(exc)

    record["git"] = collect_git_info()
    record["env"] = collect_env_info()
    record["parent_checkpoints"] = find_parent_checkpoints(cfg)

    # A stable id for cross-referencing W&B runs and eval outputs.
    commit = record["git"].get("commit_short", "nogit")
    stamp = record["timestamp"].replace(":", "").replace("-", "")[:15]
    record["run_id"] = f"{record.get('task_name', 'run')}_{commit}_{stamp}"

    return record


@rank_zero_only
def write_provenance(cfg: DictConfig, run_dir: str) -> Dict[str, Any]:
    """Write provenance.json, the uncommitted diff and the resolved config to *run_dir*.

    Returns the record so it can also be attached to checkpoints and loggers.
    """
    record = collect_provenance(cfg, run_dir=run_dir)

    try:
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, PROVENANCE_FILENAME), "w") as f:
            json.dump(record, f, indent=2, default=str)

        # The commit alone does not reproduce a dirty tree, so keep the diff too.
        if record["git"].get("is_dirty"):
            patch = _git("diff", "HEAD")
            if patch:
                with open(os.path.join(run_dir, PATCH_FILENAME), "w") as f:
                    f.write(patch)

        # Hydra saves the unresolved config; a resolved copy is what you actually ran.
        try:
            resolved = OmegaConf.to_yaml(cfg, resolve=True)
        except Exception:
            resolved = OmegaConf.to_yaml(cfg, resolve=False)
        with open(os.path.join(run_dir, RESOLVED_CONFIG_FILENAME), "w") as f:
            f.write(resolved)

        log.info(
            f"Provenance: {record['run_id']} "
            f"(git {record['git'].get('describe', 'unknown')}, "
            f"{len(record['parent_checkpoints'])} parent checkpoint(s))"
        )
    except Exception as exc:
        log.warning(f"Failed to write provenance files: {exc}")

    return record


class ProvenanceCallback(Callback):
    """Embed the run's provenance record into every checkpoint it saves.

    This is the part that survives a lost log directory: a ``.ckpt`` file carries the
    commit, config and parent checkpoints that produced it.
    """

    def __init__(self, record: Dict[str, Any]) -> None:
        super().__init__()
        self.record = record

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        try:
            checkpoint[PROVENANCE_KEY] = {
                **self.record,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
            }
        except Exception as exc:  # never block a checkpoint save
            log.warning(f"Failed to attach provenance to checkpoint: {exc}")


def describe_checkpoint(path: str, _depth: int = 0, _seen: Optional[set] = None) -> None:
    """Print a checkpoint's provenance and recursively walk its parent chain."""
    _seen = _seen if _seen is not None else set()
    indent = "  " * _depth
    real = os.path.abspath(path)
    if real in _seen:
        print(f"{indent}(cycle: {path})")
        return
    _seen.add(real)

    if not os.path.exists(path):
        print(f"{indent}{path}  [MISSING]")
        return

    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"{indent}{path}  [unreadable: {exc}]")
        return

    print(f"{indent}{path}")
    print(f"{indent}  epoch={ckpt.get('epoch')}  global_step={ckpt.get('global_step')}")

    record = ckpt.get(PROVENANCE_KEY)
    if not isinstance(record, dict):
        print(f"{indent}  no provenance record (predates provenance tracking)")
        return

    git = record.get("git", {})
    print(f"{indent}  run_id:    {record.get('run_id')}")
    print(f"{indent}  task_name: {record.get('task_name')}")
    print(f"{indent}  git:       {git.get('describe')} ({git.get('branch')})")
    if git.get("is_dirty"):
        print(f"{indent}  WARNING: working tree was dirty at run start")
    print(f"{indent}  run_dir:   {record.get('run_dir')}")
    print(f"{indent}  saved_at:  {record.get('saved_at')}")

    parents = record.get("parent_checkpoints") or []
    if parents:
        print(f"{indent}  parents:")
        for parent in parents:
            print(f"{indent}    [{parent.get('config_key')}]")
            describe_checkpoint(parent["path"], _depth=_depth + 3, _seen=_seen)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.utils.provenance <checkpoint.ckpt> [...]")
        raise SystemExit(1)
    for arg in sys.argv[1:]:
        describe_checkpoint(arg)
        print()
