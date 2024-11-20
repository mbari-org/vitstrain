# Training library for fine-tuning VIT models on custom datasets

## Installation


### Create a new environment conda
```bash
conda env create
```

### If you prefer pyenv

```bash
pyenv virtualenv 3.11.0 vittrain
pyenv activate vittrain
pip install -r requirements.txt
```

## Usage

Step 1. Download the labeled data

```bash
cd aidata
python aidata download --config $PWD/aidata/config/config_uav.yml --base-path $PWD --concepts Kelp --voc  --token $TATOR_TOKEN
```

Step 2. Crop the images

```bash
cd imagecropper
python src/run.py --image_dir $PWD/Baseline/images --output_path $PWD/Baseline/crops --data_dir $PWD/Baseline/voc --resize 224x224
```

Step 3. Train the model

Change the `model_name` , `raw_data`, and `filter_data` in the fine_tune_vit.py file to the model you want to train based on the above

```bash
python src/fine_tune_vit.py
```