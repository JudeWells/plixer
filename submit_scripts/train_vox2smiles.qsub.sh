#!/bin/bash

# Train ProFam

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N vox2smi
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
python ${ROOT_DIR}/src/train.py +experiment=train_vox2smiles_all ckpt_path=null \
data.config.batch_size=12
date