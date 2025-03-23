# echo the contents of this file
cat $0

cd /mnt/disk2/VoxelDiffOuter/VoxelDiff2
export PYTHONPATH=$(pwd):$PYTHONPATH
source ../VenvVoxelDiff/bin/activate
# run the script
python src/train.py +experiment=train_vox2smiles_combined data.num_workers=5 data.config.batch_size=2 \
ckpt_path="logs/poc2mol/runs/2025-03-08_20-45-12/checkpoints/last.ckpt" \
model.lr="1e-5" \
model.scheduler_name="cosine_with_min_lr" \
model.num_warmup_steps=2 \
model.num_training_steps=2000
