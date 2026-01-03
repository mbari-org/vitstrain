#!/bin/bash
nohup python src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--model-name mbari-m3-vits-b8 \
	--base-model /mnt/DeepSea-AI/models/M3/mbari-m3-vits-b8-20251011/ \
	--remap train_class_reduction.json \
	--add-rotations True \
	--remove-long-tail True \
	--num-epochs 30 \
	--raw-data /mnt/DeepSea-AI/data/M3/crops/ \
	--filter-data /mnt/ML_SCRATCH/M3_filtered_crops \
	--exclude-labels "marine snow" "marine organism"  > mbari-m3-vits-b8.log 2>&1 &
