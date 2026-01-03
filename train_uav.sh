#!/bin/bash
python src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--model-name mbari-uav-vits-b8 \
	--base-model facebook/dino-vitb8  \
	--num-epochs 50 \
	--add-rotations True \
	--raw-data /mnt/ProjectLibrary/901902_UAV/Downloads-AI/Nov042025_crop_classification/crops/  \
	--filter-data /mnt/ML_SCRATCH/data/uav_filter
exit
python src/fine_tune_vits.py \
	--model-name mbari-uav-vits-b-8 \
	--base-model facebook/dino-vitb8 \
	--raw-data \
	/mnt/ML_SCRATCH/UAV \
	--filter-data \
	/mnt/ML_SCRATCH/UAV/Combined \
	--add-rotations True \
	--num-epochs 5 

python src/fine_tune_vits.py \
	--model-name mbari-uav-vits-b-16 \
	--base-model google/vit-base-patch16-224-in21k \
	--raw-data \
	/mnt/ML_SCRATCH/UAV \
	--filter-data \
	/mnt/ML_SCRATCH/UAV/Combined \
	--add-rotations True \
	--num-epochs 5 
