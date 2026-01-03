#!/bin/bash
python src/fine_tune_vits.py \
	--model-name mbari-m3ctnA-vits-b16 \
	--base-model google/vit-base-patch16-224-in21k \
	--raw-data \
	/mnt/ML_SCRATCH/M3/crops \
	/mnt/ML_SCRATCH/901103-biodiversity/crops \
	--filter-data \
	/mnt/ML_SCRATCH/901103-biodiversity/combined \
	--num-epochs 50 \
	--remove-long-tail True \
	--remap train_class_reduction.json
