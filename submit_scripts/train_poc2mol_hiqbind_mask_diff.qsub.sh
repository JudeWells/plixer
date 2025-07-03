#!/bin/bash

# Train Poc2Mol on HiQBind

#$ -l tmem=127G
# -l h_vmem=64G
#$ -l gpu=true
#$ -R y
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l hostname=!(bubba*)
#$ -l h_rt=119:55:30
#$ -S /bin/bash
#$ -N DiffPoc2mol
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
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
ROOT_DIR='/SAN/orengolab/nsp13/VoxelDiffOuter/plixer/'
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
cd $ROOT_DIR
python ${ROOT_DIR}/src/train.py \
experiment=train_poc2mol_hiqbind \
ckpt_path=null \
+model.unmasking_strategy=confidence \
data.num_workers=0
date
qsub submit_scripts/train_poc2mol_hiqbind_mask_diff.qsub.sh