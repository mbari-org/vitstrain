#!/bin/bash
python src/fine_tune_vits.py \
	--early-stopping-epochs 3 \
	--model-name mbari-i2map-m3-vits-b16 \
	--base-model facebook/dino-vitb16 \
	--remove-long-tail True  \
	--num-epochs 30 \
	--raw-data /mnt/DeepSea-AI/scratch/i2mapbulk/Baseline_mbari-i2map-vits-b8-20251008-vss/crops/ /mnt/DeepSea-AI/scratch/i2map/crops/ \
	--filter-data /mnt/ML_SCRATCH/data/i2map_m3_filter-b16
