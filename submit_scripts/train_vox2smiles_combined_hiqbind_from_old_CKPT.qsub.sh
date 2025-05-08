#!/bin/bash

# Train vox2smiles combined model using a checkpoint from here on kasp:
# logs/vox2smilesZincAndPoc2MolOutputs/runs/2025-03-22_21-18-58
# original training run can be seen here: 
# https://wandb.ai/cath/voxelSmiles/runs/crz11hbc
# this run was getting 95% SMILES recovery

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -R y
#$ -l h_rt=91:55:30
#$ -S /bin/bash
#$ -N CombiResume1e4
#$ -t 1
#$ -o /SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2/qsub_logs/
#$ -wd /SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2/
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate vox
which python
ROOT_DIR='/SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2/'
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
cd $ROOT_DIR
python src/train.py \
experiment=train_vox2smiles_combined_hiqbind \
data.num_workers=0 \
data.config.batch_size=2 \
task_name="CombinedHiQBAggPropPoc2Mol" \
trainer.accumulate_grad_batches=2 \
model.config.lr=1e-4 \
ckpt_path="logs/CombinedHiQBindCkptFrmPrevCombined/runs/2025-05-06_20-51-46/checkpoints/last.ckpt" \
model.override_optimizer_on_load=True
date
