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
from typing import List
from datasets import load_dataset, DatasetDict


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def create_dataset(logger: Logger, raw_dataset_paths: List[Path], train_dataset_root: Path):
    if train_dataset_root.exists():
        logger.info(f"Removing existing dataset at {train_dataset_root}")
        shutil.rmtree(train_dataset_root)

    # Combine the raw datasets into a single dataset
    combined_stats = {}
    for path in raw_dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        # Combine the stats
        crop_path = path / 'crops'
        with open(crop_path / 'stats.json') as f:
            stats = json.load(f)
            for k,v in stats['total_labels'].items():
                if k in combined_stats:
                    combined_stats[k] += int(v)
                else:
                    combined_stats[k] = int(v)


    # Randomly downsample the dataset to 2000 images per class and copy the images to a new directory
    for label, count in combined_stats.items():
        images = []
        for path in raw_dataset_paths:
            crop_path = path / 'crops' / str(label)
            images.extend(list(crop_path.glob('*.jpg')))
        logger.info(f"Found {len(images)} images for {label}")
        if len(images) > 2000:
            images = np.random.choice(images, 2000, replace=False)
        for image in tqdm.tqdm(images, desc=f"Copying images for {label}"):
            dest = train_dataset_root / str(label) / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image, dest)

    # Load the dataset
    ds = load_dataset(train_dataset_root.as_posix())

    ds_train_test = ds['train'].train_test_split(test_size=0.2, seed=42)
    # Split the 20% test + valid in half test, half valid
    ds_valtest = ds_train_test['test'].train_test_split(test_size=0.5, seed=42)

    ds_splits = DatasetDict({
        'train': ds_train_test['train'],
        'valid': ds_valtest['train'],
        'test': ds_valtest['test']
    })

    # Create label mappings, id2label and label2id from the dataset
    id2label = {id:label for id, label in enumerate(combined_stats.keys())}
    label2id = {label:id for id,label in id2label.items()}
    logger.info(label2id)
    logger.info(id2label)
    return ds_splits, id2label, label2id
