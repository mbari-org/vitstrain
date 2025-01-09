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

Step 1. Download the labeled data and crop the images using the [aidata repository](https://github.com/mbari-org/aidata)

```bash
cd aidata
python aidata download \
        --config $PWD/aidata/config/config_uav.yml \
        --base-path $PWD --voc \
        --token $TATOR_TOKEN --crop-roi --resize 224
```

Step 2. Train the model

```bash
python src/fine_tune_vit.py \
        --data-path $PWD/Baseline \
        --base-model google/vit-base-patch16-224-in21k
        --model-name mbari-uav-vit-b-16 \
        --epochs 30
```

![docs/imgs/confusion_matrix.png](./docs/imgs/confusion_matrix.png)
![docs/imgs/loss_curve.png](./docs/imgs/loss_curve.png)

last updated: 2025-01-09
