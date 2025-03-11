# echo the contents of this file
cat $0

cd /mnt/disk2/VoxelDiffOuter/VoxelDiff2
export PYTHONPATH=$(pwd):$PYTHONPATH
source ../VenvVoxelDiff/bin/activate
# run the script
python src/models/train.py +experiment=train_poc2mol_plinder data.num_workers=8 data.config.batch_size=2