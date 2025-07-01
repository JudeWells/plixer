# echo the contents of this file
cat $0

cd /mnt/disk2/VoxelDiffOuter/VoxelDiff2
export PYTHONPATH=$(pwd):$PYTHONPATH
source ../VenvVoxelDiff/bin/activate
# run the script
python src/train.py +experiment=train_poc2mol data.num_workers=2 data.config.batch_size=2 \
ckpt_path="logs/poc2mol_PDBbind/runs/2025-03-23_08-02-50/checkpoints/last.ckpt" \
model.lr="1e-4" \
model.num_warmup_steps=100 \
+model.num_training_steps=800 \
model.override_optimizer_on_load=true \
trainer.accumulate_grad_batches=64 \
trainer.max_epochs=33800
