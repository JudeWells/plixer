#!/usr/bin/env python
"""Print the provenance of one or more checkpoints, walking their parent lineage.

Answers "which code, config and parent model produced this .ckpt?" for any checkpoint
saved since provenance tracking was added. Checkpoints predating it report that fact
rather than failing.

    python scripts/describe_checkpoint.py logs/poc2mol/runs/*/checkpoints/last.ckpt
"""

import sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.provenance import describe_checkpoint  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for path in sys.argv[1:]:
        describe_checkpoint(path)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
