#!/bin/bash

# Train ProFam

#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=23:55:30
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
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
python ${ROOT_DIR}/src/train.py data=af50_single_seq trainer=gpu logger=wandb
date