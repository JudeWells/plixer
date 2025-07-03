#!/bin/bash

# Train ProFam

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -R y
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N vox2smiCkpt2
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
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ROOT_DIR='/SAN/orengolab/nsp13/VoxelDiffOuter/plixer/'
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
cd $ROOT_DIR
python src/train.py \
experiment=train_vox2smiles_combined_hiqbind \
data.num_workers=0 \
ckpt_path=checkpoints/combined_protein_to_smiles/epoch_000.ckpt
date
