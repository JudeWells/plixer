#!/bin/bash

# Train ProFam

#$ -l tmem=7G
#$ -l h_vmem=7G
#$ -l h_rt=6:55:30
#$ -S /bin/bash
#$ -N ImgGen
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
python ${ROOT_DIR}/scripts/adhoc_analysis/visualise_PROTEIN_ONLY_in_test.py --task_idx $(($SGE_TASK_ID - 1)) --num_tasks $SGE_TASK_LAST
date