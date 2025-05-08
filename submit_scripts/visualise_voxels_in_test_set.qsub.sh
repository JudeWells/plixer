#!/bin/bash

# Train ProFam

#$ -l tmem=3.9G
#$ -l h_vmem=3.9G
#$ -l h_rt=7:55:30
#$ -S /bin/bash
#$ -N ImgGen2
#$ -t 80-300
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
python ${ROOT_DIR}/scripts/adhoc_analysis/visualise_voxels_in_test_set.py --task_idx $(($SGE_TASK_ID - 1)) --num_tasks $SGE_TASK_LAST
date
