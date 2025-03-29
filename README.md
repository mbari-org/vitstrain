# Training library for fine-tuning ViTs (Vision Transformer) models on custom datasets

## Installation ⚙️ 


### Create a new environment conda
```bash
conda env create
conda activate vitstrain
```

### If you prefer pyenv

```bash
pyenv virtualenv 3.11.0 vitstrain
pyenv activate vitstrain
pip install -r requirements.txt
```

## Training 🚀  

Step 1. Download the labeled data and crop the images using the [mbari-aidata pip module](https://github.com/mbari-org/aidata)
 
Data should be in folder per class with and required stats.json file. 
For example, the folder structure should look like this:

```
└── crops
    ├── cats
    │   ├── cat.0.jpg
    │   ├── cat.1.jpg
    │   ├── cat.10.jpg
    │   ├── cat.100.jpg 
    ├── dogs
    │   ├── dog.0.jpg
    │   ├── dog.1.jpg
    │   ├── dog.10.jpg
    │   ├── dog.100.jpg 
    └── stats.json
```                                                                                                                                                                                          

The stats.json file should contain the following information:

```json
{ 
    "total_labels": {
        "cats": 100,
        "dogs": 100
    }
}
```

Here, we are using the `config_uav.yml` configuration file to download the UAV dataset,
download the data, crop the images, and resize them to 224x224 pixels.
TODO: add more details about the configuration file.

```bash
pip install mbari-aidata
cd aidata
python aidata download \
        --config config_uav.yml \
        --base-path $PWD  \
        --version Baseline \
        --token $TATOR_TOKEN --crop-roi --resize 224
```

Step 2. Train the model

```bash
python src/fine_tune_vit.py \
        --data-path $PWD/Baseline/crops \
        --base-model google/vit-base-patch16-224-in21k
        --model-name mbari-uav-vit-b16 \
        --epochs 30
```

Example output:
```text
/Volumes/DeepSea-AI/models/UAV/mbari-uav-vit-b16-20250108/
├── all_results.json
├── checkpoint-1710
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── preprocessor_config.json
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── config.json
├── confusion_matrix_mbari-uav-vit-b16-20250108_2025-01-08 073852.png
├── eval_results.json
├── loss_curve_mbari-uav-vit-b16-20250108_2025-01-08_073852.png
├── model.safetensors
├── preprocessor_config.json
└── training_args.bin
```

To remap the classes, use the `--remap` flag, passing in a file with a json formatted dictionary


```json

{
    "oldname" : "newname"
}
```

For example

```json
{
    "cats" : "felines",
    "dogs" : "canines"
}
```

THen a

```bash
python src/fine_tune_vit.py \
        ...
        --remap remap.json
```

![docs/imgs/confusion_matrix.png](./docs/imgs/confusion_matrix.png)
![docs/imgs/loss_curve.png](./docs/imgs/loss_curve.png)

last updated: 2025-03-29
