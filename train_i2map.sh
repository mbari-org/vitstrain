#!/bin/bash

rm *.json

python src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--model-name mbari-i2map-vits-b8 \
	--base-model facebook/dino-vitb8 \
	--num-epochs 50 --add-rotations True \
	--raw-data /mnt/DeepSea-AI/scratch/i2mapbulk/Baseline_mbari-i2map-vits-b8-20251008-vss/crops/  \
	--filter-data /mnt/ML_SCRATCH/data/i2map_filter

exit
python src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--remove-long-tail True \
	--model-name mbari-i2map-vits-b32nt \
       	--base-model openai/clip-vit-base-patch32 \
	--raw-data \
	/mnt/ML_SCRATCH/i2map/Baseline/crops \
	/mnt/ML_SCRATCH/i2mapbulk/crops \
	--filter-data \
	/mnt/ML_SCRATCH/i2map/Combined \
	--num-epochs 30

python ~/code/vittrainclean/src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--remove-long-tail True \
        --model-name mbari-i2map-vits-b16nt \
	--base-model google/vit-base-patch16-224-in21k \
	--raw-data \
	/mnt/ML_SCRATCH/i2map/Baseline/crops \
	/mnt/ML_SCRATCH/i2mapbulk/crops \
	--filter-data \
	/mnt/ML_SCRATCH/i2map/Combined \
	--num-epochs 30

python ~/code/vittrainclean/src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--remove-long-tail True \
	--model-name mbari-i2map-vits-b8nt \
	--base-model facebook/dino-vitb8 \
	--raw-data \
	/mnt/ML_SCRATCH/i2map/Baseline/crops \
	/mnt/ML_SCRATCH/i2mapbulk/crops \
	--filter-data \
	/mnt/ML_SCRATCH/i2map/Combined \
	--num-epochs 30

python src/fine_tune_vits.py \
	--remove-long-tail True \
	--model-name mbari-i2map-vits-b-8ntnr \
	--base-model /mnt/DeepSea-AI/models/i2MAP/mbari-i2map-vits-b-8-20250109 \
	--raw-data \
	/mnt/ML_SCRATCH/i2map \
	/mnt/ML_SCRATCH/i2mapbulk/ \
	--filter-data \
	/mnt/ML_SCRATCH/i2map/Combined \
	--num-epochs 30

