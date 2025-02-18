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


def compute_mean_std(dataset):
    from PIL import ImageStat, Image
    ds_mean = dataset.map(lambda x: {
                         "mean": ImageStat.Stat(x["image"]).mean},
                         remove_columns=dataset.column_names,
                         num_proc=16)
    ds_std = dataset.map(lambda x: {
                         "stddev": ImageStat.Stat(x["image"]).stddev},
                         remove_columns=dataset.column_names,
                         num_proc=16)

    avg_mean = np.zeros(3)
    avg_std = np.zeros(3)

    total = len(ds_mean)
    for i in range(total):
        avg_mean += np.array(ds_mean[i]["mean"])
        avg_std += np.array(ds_std[i]["stddev"])

    avg_mean /= total
    avg_std /= total

    # Normalize to 0-1 to match the range in the Transformer processor output
    avg_mean /= 255
    avg_std /= 255
    return list(avg_mean), list(avg_std)

def create_dataset(logger: Logger, remove_long_tail:bool, raw_dataset_paths: List[Path], train_dataset_root: Path):
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
        stats_path = crop_path / 'stats.json'
        if not stats_path.exists():
            raise FileNotFoundError(f"Path {stats_path} does not exist")

        with stats_path.open() as f:
            stats = json.load(f)
            for k,v in stats['total_labels'].items():
                if k in combined_stats:
                    combined_stats[k] += int(v)
                else:
                    combined_stats[k] = int(v)

    # Copy the images to a new directory and revise the stats in case there are errors
    correct_stats = {}
    for label, count in combined_stats.items():
        images = []
        for path in raw_dataset_paths:
            crop_path = path / 'crops' / str(label)
            images.extend(list(crop_path.glob('*.jpg')))
        logger.info(f"Found {len(images)} images for {label}")
        if len(images) > 0:
            correct_stats[label] = len(images)
        for image in tqdm.tqdm(images, desc=f"Copying images for {label}"):
            dest = train_dataset_root / str(label) / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image, dest)
    combined_stats = correct_stats

    deleted_labels = {}
    if remove_long_tail:
        # This is to avoid overfitting on labels with very few examples
        # Count the number of images in each label and remove labels with less than 10 images
        revised_stats = {}
        for d in train_dataset_root.iterdir():
            if d.is_dir():
                count = len(list(d.glob('*')))
                if count < 10:
                    logger.info(f"Removing label {d.name} with {count} images")
                    shutil.rmtree(d)
                    deleted_labels[d.name] = count
                else:
                    logger.info(f"Keeping label {d.name} with {count} images")
                    revised_stats[d.name] = count
        combined_stats = revised_stats

    with (train_dataset_root / 'stats.json').open('w') as f:
        json.dump(combined_stats, f)

    # Write the deleted labels to a json file
    with (train_dataset_root / 'deleted_labels.json').open('w') as f:
        json.dump(deleted_labels, f)

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
    id2label = {id:label for id, label in enumerate(sorted(combined_stats.keys()))}
    label2id = {label:id for id,label in id2label.items()}
    logger.info(label2id)
    logger.info(id2label)

    # Compute the mean and std of the training dataset
    mean, std = compute_mean_std(ds_splits["train"])

    logger.info(f'Number of training samples: {len(ds_splits["train"])}')
    logger.info(f'Number of validation samples: {len(ds_splits["valid"])}')
    logger.info(f'Number of test samples: {len(ds_splits["test"])}')
    logger.info(f'Mean: {mean}')
    logger.info(f'Std: {std}')

    return ds_splits, id2label, label2id, mean, std
