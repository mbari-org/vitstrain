# vitstrain
# Filename: src/data_utils.py
# Description: Data utilities for preparing image datasets for training Vision Transformer models

import shutil
from logging import Logger
from pathlib import Path
from PIL import ImageStat, Image
import tqdm
import json
import numpy as np
import torch
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_mean_std(dataset):
    ds_mean = dataset.map(lambda x: {
        "mean": ImageStat.Stat(x["image"]).mean},
                          remove_columns=dataset.column_names,
                          keep_in_memory=False,
                          num_proc=16)
    ds_std = dataset.map(lambda x: {
        "stddev": ImageStat.Stat(x["image"]).stddev},
                         remove_columns=dataset.column_names,
                         keep_in_memory=False,
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


def create_dataset(logger: Logger, remove_long_tail: bool, raw_dataset_paths: List[Path], train_dataset_root: Path,
                   remap_class: Dict[str, str] = None, exclude_labels: List[str] = None,
                   min_images_per_class: int = 10):
    if train_dataset_root.exists():
        logger.info(f"Removing existing dataset at {train_dataset_root}")
        shutil.rmtree(train_dataset_root)

    # Combine the raw dataset stats
    combined_stats = {}
    for path in raw_dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        # Combine the stats
        stats_path = path / 'stats.json'
        if not stats_path.exists():
            raise FileNotFoundError(f"Path {stats_path} does not exist")

        with stats_path.open() as f:
            stats = json.load(f)
            for k, v in stats['total_labels'].items():
                if k in combined_stats:
                    combined_stats[k] += int(v)
                else:
                    combined_stats[k] = int(v)

    # Remap the classes if necessary, then copy the images to a new directory and
    # revise the stats in case there are errors in the original stats.json
    correct_stats = {}
    excluded_labels = {}
    for label, count in combined_stats.items():
        # Skip excluded labels
        if exclude_labels is not None and label in exclude_labels:
            logger.info(f"Excluding label {label} with {count} images")
            excluded_labels[label] = count
            continue

        images = []
        final_label = label
        if remap_class is not None:
            if label in remap_class.keys():
                final_label = remap_class[label]
        correct_stats[final_label] = 0
        for path in raw_dataset_paths:
            class_path = path / str(label)
            images.extend(list(class_path.glob('*.jpg')))
            images.extend(list(class_path.glob('*.png')))
        logger.info(f"Found {len(images)} images for {label} mapped to {final_label}")
        for image in tqdm.tqdm(images, desc=f"Copying images for {label} to {final_label}"):
            dest = train_dataset_root / str(final_label) / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image, dest)
            correct_stats[final_label] += 1
    combined_stats = correct_stats

    deleted_labels = {}
    if remove_long_tail:
        # This is to avoid overfitting on labels with very few examples
        # Count the number of images in each label and remove labels with less than min_images_per_class images
        revised_stats = {}
        for d in train_dataset_root.iterdir():
            if d.is_dir():
                count = len(list(d.glob('*')))
                if count < min_images_per_class:
                    logger.info(f"Removing label {d.name} with {count} images")
                    shutil.rmtree(d)
                    deleted_labels[d.name] = count
                else:
                    logger.info(f"Keeping label {d.name} with {count} images")
                    revised_stats[d.name] = count
        combined_stats = revised_stats

    # Using PIL, augment by random cropping with overlap for all classes with fewer than min_images_per_class examples
    # so that there are at least min_images_per_class examples per class
    for d in train_dataset_root.iterdir():
        if d.is_dir():
            count = len(list(d.glob('*')))
            if count < min_images_per_class and count > 0:
                images = list(d.glob('*'))
                augment_count = min_images_per_class - count
                logger.info(f"Augmenting label {d.name} with {count} images to {count + augment_count} images")
                for i in range(augment_count):
                    image_path = images[i % count]
                    image = Image.open(image_path)
                    width, height = image.size
                    # Random crop with 80% overlap
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    left = np.random.randint(0, width - new_width + 1)
                    top = np.random.randint(0, height - new_height + 1)
                    right = left + new_width
                    bottom = top + new_height
                    cropped_image = image.crop((left, top, right, bottom))
                    augmented_image_path = d / f"{image_path.stem}_aug_{i}{image_path.suffix}"
                    cropped_image.save(augmented_image_path)
                    combined_stats[d.name] += 1

    with (train_dataset_root / 'stats.json').open('w') as f:
        json.dump(combined_stats, f)

    # Write the deleted labels (from long-tail removal) and excluded labels to json files
    all_removed_labels = {**deleted_labels, **excluded_labels}
    with (train_dataset_root / 'deleted_labels.json').open('w') as f:
        json.dump(all_removed_labels, f)

    # Load the dataset
    logger.info(f"Loading dataset {train_dataset_root}...")
    ds = load_dataset(train_dataset_root.as_posix())

    logger.info(f"Splitting data")

    # Convert to pandas dataframe
    df_train = ds['train'].to_pandas()

    X = df_train['image']
    y = df_train['label']

    # Using sklearn train_test_split instead of huggingface because it has the stratify flag.
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Split the 20% test + valid in half test, half valid
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42,
                                                    stratify=y_test_val)

    ds_splits = DatasetDict({
        'train': ds['train'].select(X_train.index),
        'valid': ds['train'].select(X_val.index),
        'test': ds['train'].select(X_test.index)
    })

    # Create label mappings, id2label and label2id from the dataset
    logger.info(f"Creating label maps and computing statistics")
    id2label = {id: label for id, label in enumerate(sorted(combined_stats.keys()))}
    label2id = {label: id for id, label in id2label.items()}
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

