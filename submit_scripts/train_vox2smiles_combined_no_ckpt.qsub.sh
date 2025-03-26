#!/bin/bash

# Train ProFam

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N vox2smiNoCkpt2
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
+experiment=train_vox2smiles_combined \
data.num_workers=0 \
data.config.batch_size=2 \
trainer.val_check_interval=5000 \
task_name="vox2smilesZincAndPoc2MolOutputsNoCkpt" \
ckpt_path="/SAN/orengolab/nsp13/VoxelDiffOuter/VoxelDiff2/logs/vox2smilesZincAndPoc2MolOutputsNoCkpt/runs/2025-03-23_19-59-20/checkpoints/interrupted.ckpt"
date
