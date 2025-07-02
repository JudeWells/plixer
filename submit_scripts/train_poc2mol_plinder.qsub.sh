#!/bin/bash

# Train Poc2Mol on plinder

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=93:55:30
#$ -S /bin/bash
#$ -N poc2molPlin2
#$ -t 1
#$ -o /SAN/orengolab/nsp13/VoxelDiffOuter/plixer/qsub_logs/
#$ -wd /SAN/orengolab/nsp13/VoxelDiffOuter/plixer/
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate vox
which python
ROOT_DIR='/SAN/orengolab/nsp13/VoxelDiffOuter/plixer/'
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
cd $ROOT_DIR
python ${ROOT_DIR}/src/train.py \
+experiment=train_poc2mol_plinder \
task_name="poc2molPlinder" \
ckpt_path="/SAN/orengolab/nsp13/VoxelDiffOuter/plixer/logs/poc2molPlinder/runs/2025-03-23_22-39-56/checkpoints/last.ckpt" \
data.config.batch_size=2 \
model.override_optimizer_on_load=true \
model.lr="5e-5"

date
