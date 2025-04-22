#!/bin/bash

# Hyper‑parameter optimisation sweep for Poc2Mol (HiQBind) using Optuna
# Each SGE array task runs an independent Optuna worker that connects
# to the same shared study backed by an SQLite file on the shared FS.
# Adjust resource requests as appropriate for your cluster.

#$ -l tmem=64G            # GPU memory requirement per task
#$ -l gpu=true            # Request any available GPU
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=47:55:00       # Max run‑time (can be lower/higher as needed)
#$ -S /bin/bash
#$ -N poc2molOptuna
#$ -cwd                   # Use current working directory
#$ -t 1-16                # Number of parallel Optuna workers (array size)
#$ -o /SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2/qsub_logs/
#$ -j y                   # Merge stdout & stderr

# Diagnostic info
hostname
nvidia-smi || true

echo "### Starting Optuna worker $SGE_TASK_ID at $(date) ###"

# Activate environment (adapt to your setup)
conda activate vox

# Make sure PYTHONPATH points to project root to allow local imports
ROOT_DIR='/SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2'
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
cd "$ROOT_DIR"

# Shared Optuna SQLite DB lives on shared filesystem so that all workers
# can cooperate.  Feel free to change the path.
STORAGE="sqlite:///optuna_poc2mol_hiqbind.db"

python scripts/tune_poc2mol_optuna.py \
  --experiment train_poc2mol_hiqbind \
  --study_name poc2mol_hiqbind_optuna \
  --storage "$STORAGE" \
  --n_trials 20 \
  --max_epochs 100

echo "### Finished worker $SGE_TASK_ID at $(date) ###" 