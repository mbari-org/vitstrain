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

### Step 1. Download the labeled data 


TL;DR: For a quick-start, use the included `data/catsdogs.tar.gz` file, which contains data in the required format.

```bash
tar -xvzf data/catsdogs.tar.gz
```

To download and crop data using the `mbari-aidata` package, see more detailed documentation in our
[aidata documentation](https://docs.mbari.org/internal/ai/classification-training/#training-a-classification-model).

 
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

### Step 2. Train the model

```bash
python src/fine_tune_vits.py \
        --raw-data $PWD/data/crops \
        --base-model google/vit-base-patch16-224-in21k
        --model-name catsdogs-vit-b16 \
        --num-epochs 5
```

To skip data preparation and train from an existing prepared dataset, use `--train-only`:

```bash
python src/fine_tune_vits.py \
        --train-only \
        --filter-data $PWD/data/data_filter \
        --base-model google/vit-base-patch16-224-in21k \
        --model-name catsdogs-vit-b16 \
        --num-epochs 5
```

`--filter-data` must point to a previously prepared dataset directory.

To also export the trained model to ONNX, add `--export-onnx`:

```bash
python src/fine_tune_vits.py \
        --raw-data $PWD/data/crops \
        --base-model google/vit-base-patch16-224-in21k \
        --model-name catsdogs-vit-b16 \
        --num-epochs 5 \
        --export-onnx
```

This writes `model.onnx` into the model output directory. The graph takes a `pixel_values`
input at the resolution the model was trained at, with a dynamic batch dimension, and returns
`logits`. After exporting, the outputs are compared against PyTorch and the difference is
reported in the training log.

To add rebalanced supervised contrastive (SupCon) loss on backbone embeddings — useful for
long-tailed datasets — pass `--use-supcon`. This is combined with the existing focal
classification loss so the classifier head still trains:

```bash
python src/fine_tune_vits.py \
        --raw-data $PWD/data/crops \
        --base-model google/vit-base-patch16-224-in21k \
        --model-name catsdogs-vit-b16 \
        --num-epochs 5 \
        --use-supcon \
        --supcon-temperature 0.07 \
        --supcon-weight 1.0
```

Rare classes are up-weighted so their embeddings are pulled tighter. Larger batches give
SupCon more same-class pairs; gradient accumulation does not enlarge the contrastive batch.

Example output (`model.onnx` only when `--export-onnx` is used):
```text
catsdogs-vit-b16-20250828
├── all_results.json
├── checkpoint-100
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── preprocessor_config.json
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── config.json
├── confusion_matrix_catsdogs-vit-b16-20250828_2025-08-28_144843.png
├── eval_results.json
├── loss_curve_catsdogs-vit-b16-20250828_2025-08-28_144843.png
├── model.onnx
├── model.safetensors
├── optimal_thresholds_catsdogs-vit-b16-20250828_20250828_144843.csv
├── per_class_metrics.csv
├── pr_curves_catsdogs-vit-b16-20250828_2025-08-28_144843.png
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

Then a

```bash
python src/fine_tune_vit.py \
        ...
        --remap remap.json
```

![docs/imgs/confusion_matrix.png](./docs/imgs/confusion_matrix.png)
![docs/imgs/loss_curve.png](./docs/imgs/loss_curve.png)
![docs/imgs/pr_curves.png](./docs/imgs/pr_curves.png)

last updated: 2026-09-13
