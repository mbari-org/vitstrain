#!/bin/bash

python src/fine_tune_vits.py \
	--early-stopping-epochs 3 \
	--model-name mbari-ifcb2014-vitb16 \
	--base-model google/vit-base-patch16-224-in21k  \
	--raw-data \
	/mnt/ML_SCRATCH/ifcb/raw/2006-square \
	/mnt/ML_SCRATCH/ifcb/raw/2007-square \
	/mnt/ML_SCRATCH/ifcb/raw/2008-square \
	/mnt/ML_SCRATCH/ifcb/raw/2009-square \
	/mnt/ML_SCRATCH/ifcb/raw/2010-square \
	/mnt/ML_SCRATCH/ifcb/raw/2011-square \
	/mnt/ML_SCRATCH/ifcb/raw/2012-square \
	/mnt/ML_SCRATCH/ifcb/raw/2013-square \
	/mnt/ML_SCRATCH/ifcb/raw/2014-square \
	--filter-data \
	/mnt/ML_SCRATCH/ifcb/processed \
	--num-epochs 50

