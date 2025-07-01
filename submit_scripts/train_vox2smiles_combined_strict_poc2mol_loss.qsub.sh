#!/bin/bash

# Train ProFam

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -R y
#$ -l h_rt=119:55:30
#$ -S /bin/bash
#$ -N Combi0p68
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
ckpt_path=checkpoints/combined_protein_to_smiles/epoch_000.ckpt \
data.train_dataset.prob_poc2mol=0.8 \
data.train_dataset.max_poc2mol_loss=0.68 \
trainer.accumulate_grad_batches=32
date
