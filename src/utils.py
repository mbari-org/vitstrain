# vittrain
# Filename: src/utils.py
# Description: Utilities used when fine-tuning a Vision Transformer model with HuggingFace

import shutil
from logging import Logger
from pathlib import Path
import tqdm
import json
import numpy as np
import torch
from datasets import load_dataset, DatasetDict


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def create_dataset(logger: Logger, raw_dataset_path: Path, train_dataset_root: Path):
    if train_dataset_root.exists():
        logger.info(f"Removing existing dataset at {train_dataset_root}")
        shutil.rmtree(train_dataset_root)

    crop_path = raw_dataset_path / 'crops'
    labels = crop_path / 'stats.json'
    with open(labels) as f:
        stats = json.load(f)

    total_labels = {k:int(v) for k,v in stats['total_labels'].items()}

    # Randomly downsample the dataset to 2000 images per class and copy the images to a new directory
    for label, count in total_labels.items():
        images = list((crop_path / str(label)).rglob('*.jpg'))
        if len(images) > 2000:
            images = np.random.choice(images, 2000, replace=False)
        for image in tqdm.tqdm(images, desc=f"Copying images for {label}"):
            dest = train_dataset_root / str(label) / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image, dest)

    # Load the dataset
    ds = load_dataset(train_dataset_root.as_posix())

    ds_train_devtest = ds['train'].train_test_split(test_size=0.2, seed=42)
    ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)

    ds_splits = DatasetDict({
        'train': ds_train_devtest['train'],
        'valid': ds_devtest['train'],
        'test': ds_devtest['test']
    })

    train_ds = ds_splits['train']
    # Create label mappings, id2label and label2id from the dataset
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}
    logger.info(label2id)
    logger.info(id2label)
    return ds_splits, id2label, label2id
