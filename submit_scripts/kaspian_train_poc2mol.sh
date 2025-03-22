# echo the contents of this file
cat $0

cd /mnt/disk2/VoxelDiffOuter/VoxelDiff2
export PYTHONPATH=$(pwd):$PYTHONPATH
source ../VenvVoxelDiff/bin/activate
# run the script
python src/train.py +experiment=train_poc2mol_plinder data.num_workers=5 data.config.batch_size=2 \
ckpt_path="logs/poc2mol_PDBbind/runs/2025-03-19_22-50-16/checkpoints/last.ckpt"
