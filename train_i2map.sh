#!/bin/bash
set -euo pipefail

# Plus-shaped SupCon sweep: vary weight at T=0.07, vary temperature at W=0.3.
# First run prepares --filter-data; later runs reuse it with --train-only.
FILTER_DATA=/mnt/ML_SCRATCH/i2MAPsupcon
SWEEPS=(
	"0.07 0.3"
	"0.07 0.1"
	"0.07 1.0"
	"0.10 0.3"
	"0.20 0.3"
)

train_only=()
for spec in "${SWEEPS[@]}"; do
	read -r temp weight <<< "$spec"
	python src/fine_tune_vits.py \
		--early-stopping-epochs 3 \
		--model-name "mbari-i2map-dinov3-supcon-t${temp}-w${weight}" \
		--base-model /mnt/DeepSea-AI/models/facebook/dinov3-vitl16-pretrain-lvd1689m/ \
		--num-epochs 30 \
		--add-rotations True \
		--use-supcon \
		--supcon-temperature "$temp" \
		--supcon-weight "$weight" \
		--export-onnx \
		"${train_only[@]}" \
		--raw-data \
		/mnt/ML_SCRATCH/i2map/videos/2-stage/crops \
		/mnt/ML_SCRATCH/i2map/Baseline/crops \
		/mnt/ML_SCRATCH/i2MAPaug/crops/ \
		--exclude-labels "marine organism" "Unknown" "Teuthida" \
		--filter-data "$FILTER_DATA"
	train_only=(--train-only)
done

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

