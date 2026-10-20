# vitstrain
# Filename: src/data_utils.py
# Description: Data utilities for preparing image datasets for training Vision Transformer models

import re
import shutil
from logging import Logger
from pathlib import Path
from PIL import ImageStat, Image
import tqdm
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict, Image as ImageFeature
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Optional


def _image_value_to_relpath(image_value, dataset_root: Path) -> str:
    """Extract a stable dataset-root-relative file path from an image column value."""
    image_path = None
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
    elif isinstance(image_value, (str, Path)):
        image_path = image_value
    elif hasattr(image_value, "filename"):
        image_path = getattr(image_value, "filename")

    if not image_path:
        return str(image_value)

    image_path = Path(image_path)
    try:
        return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except Exception:
        # Fall back to class/file style when an absolute root-relative path is unavailable.
        return image_path.as_posix()


def build_label_maps(logger: Logger, ds: DatasetDict, combined_stats: Dict[str, int]):
    """Build id2label/label2id from the class ordering the imagefolder loader assigned.

    The integer labels in the dataset come from the directories on disk, so deriving the maps
    from stats.json instead shifts every id whenever the two disagree.
    """
    class_names = list(ds["train"].features["label"].names)

    stats_only = sorted(set(combined_stats) - set(class_names))
    disk_only = sorted(set(class_names) - set(combined_stats))
    if stats_only or disk_only:
        logger.warning(
            f"stats.json ({len(combined_stats)} labels) and the dataset directories "
            f"({len(class_names)} labels) disagree; using the directory ordering. "
            f"In stats.json only: {stats_only}. On disk only: {disk_only}."
        )

    id2label = {id: label for id, label in enumerate(class_names)}
    label2id = {label: id for id, label in id2label.items()}
    return id2label, label2id


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    batch = {"pixel_values": pixel_values, "labels": labels}
    if "depth" in examples[0]:
        # NaN marks an unknown depth; the model turns it into a "missing" flag
        batch["depth"] = torch.tensor([example["depth"] for example in examples], dtype=torch.float32)
    return batch


# Augmented copies made in create_dataset are named <stem>_aug_<i>
AUG_SUFFIX_RE = re.compile(r"_aug_\d+$")


def load_depth_table(depth_csv: Path) -> Dict[str, float]:
    """Reads a CSV with 'roi' and 'depth' columns into a {roi filename stem: depth in meters} lookup."""
    df = pd.read_csv(depth_csv)
    missing_cols = {"roi", "depth"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{depth_csv} is missing required column(s): {sorted(missing_cols)}")
    stems = df["roi"].astype(str).map(lambda r: Path(r).stem)
    return dict(zip(stems, df["depth"].astype(float)))


def attach_depth(logger: Logger, ds_splits: DatasetDict, depth_csv: Path) -> DatasetDict:
    """
    Adds a float 'depth' column to every split, looked up by ROI filename stem.
    ROIs not in the table (or with a blank depth) get NaN, i.e. unknown depth.
    """
    depth_by_roi = load_depth_table(depth_csv)
    logger.info(f"Loaded depth for {len(depth_by_roi)} ROIs from {depth_csv}")

    out = {}
    for split, ds in ds_splits.items():
        # Read only the file paths; decoding every image just to get its name is slow
        paths = ds.cast_column("image", ImageFeature(decode=False))["image"]
        depths = []
        for p in paths:
            stem = AUG_SUFFIX_RE.sub("", Path(p["path"]).stem)
            depths.append(float(depth_by_roi.get(stem, np.nan)))
        n_known = int(np.sum(~np.isnan(depths)))
        logger.info(f"{split}: depth known for {n_known}/{len(depths)} ROIs")
        out[split] = ds.add_column("depth", depths)
    return DatasetDict(out)


def depth_stats(ds: Dataset) -> tuple[float, float]:
    """Mean and std of the known depths in a split, used to normalize depth inside the model."""
    d = np.array(ds["depth"], dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        raise ValueError("No ROI in the training split has a known depth; check the --depth-csv 'roi' values.")
    return float(d.mean()), float(max(d.std(), 1e-6))


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
                   min_images_per_class: int = 10, split_json_path: Optional[Path] = None):
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
        correct_stats.setdefault(final_label, 0)
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

    # Labels with no images on disk get no directory, so they must not end up in the label maps
    empty_labels = sorted(label for label, count in correct_stats.items() if count == 0)
    if empty_labels:
        logger.warning(
            f"Dropping {len(empty_labels)} label(s) listed in stats.json with no .jpg/.png images "
            f"in the raw data: {empty_labels}"
        )
    combined_stats = {label: count for label, count in correct_stats.items() if count > 0}

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
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
    )

    train_indices = list(map(int, X_train.index.tolist()))
    val_indices = list(map(int, X_val.index.tolist()))
    test_indices = list(map(int, X_test.index.tolist()))

    index_to_file = {
        int(idx): _image_value_to_relpath(df_train.at[idx, "image"], train_dataset_root)
        for idx in df_train.index.tolist()
    }
    train_files = [index_to_file[idx] for idx in train_indices]
    val_files = [index_to_file[idx] for idx in val_indices]
    test_files = [index_to_file[idx] for idx in test_indices]

    if split_json_path is not None:
        split_json_path.parent.mkdir(parents=True, exist_ok=True)
        split_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": train_dataset_root.as_posix(),
            "splitting": {
                "method": "sklearn.model_selection.train_test_split",
                "random_state": 42,
                "test_size_total": 0.2,
                "val_fraction_of_test_val": 0.5,
                "stratify": True,
            },
            "counts": {
                "train": len(train_indices),
                "valid": len(val_indices),
                "test": len(test_indices),
            },
            "files": {
                "train": train_files,
                "valid": val_files,
                "test": test_files,
            },
        }
        logger.info(f"Saving (overwriting) persisted split to {split_json_path}")
        with split_json_path.open("w") as f:
            json.dump(split_payload, f, indent=2, sort_keys=True)

    ds_splits = DatasetDict({
        'train': ds['train'].select(train_indices),
        'valid': ds['train'].select(val_indices),
        'test': ds['train'].select(test_indices)
    })

    # Create label mappings, id2label and label2id from the dataset
    logger.info(f"Creating label maps and computing statistics")
    id2label, label2id = build_label_maps(logger, ds, combined_stats)
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


def load_prepared_dataset(
    logger: Logger,
    train_dataset_root: Path,
    *,
    split_json_path: Optional[Path] = None,
) -> tuple[DatasetDict, Dict[int, str], Dict[str, int], List[float], List[float]]:
    """
    Load an already-prepared imagefolder dataset rooted at train_dataset_root.

    If split_json_path exists and contains the persisted file lists, the same split is reloaded.
    Otherwise, a new stratified split is computed from the currently-present files.
    """
    if not train_dataset_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset root {train_dataset_root} does not exist. "
            f"Either run without --train-only to build it, or point --filter-data to an existing prepared dataset."
        )

    stats_path = train_dataset_root / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset is missing {stats_path}. "
            f"Expected a directory created by the data prep step."
        )

    logger.info(f"Loading prepared dataset {train_dataset_root}...")
    ds = load_dataset(train_dataset_root.as_posix())
    df_train = ds["train"].to_pandas()

    # Build stable relpaths for index mapping
    index_to_file = {
        int(idx): _image_value_to_relpath(df_train.at[idx, "image"], train_dataset_root)
        for idx in df_train.index.tolist()
    }
    file_to_index: Dict[str, int] = {rel: idx for idx, rel in index_to_file.items()}

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

    used_persisted_split = False
    if split_json_path is not None and split_json_path.exists():
        try:
            with split_json_path.open() as f:
                payload = json.load(f)
            files = payload.get("files") or {}
            train_files = files.get("train") or []
            val_files = files.get("valid") or []
            test_files = files.get("test") or []

            train_indices = [file_to_index[p] for p in train_files if p in file_to_index]
            val_indices = [file_to_index[p] for p in val_files if p in file_to_index]
            test_indices = [file_to_index[p] for p in test_files if p in file_to_index]

            if train_indices and val_indices and test_indices:
                used_persisted_split = True
                logger.info(f"Reusing persisted split from {split_json_path}")
            else:
                logger.warning(
                    f"Persisted split at {split_json_path} did not match current files; recomputing split."
                )
        except Exception as e:
            logger.warning(f"Failed to load persisted split from {split_json_path}: {e}. Recomputing split.")

    if not used_persisted_split:
        logger.info("Computing new stratified train/valid/test split from prepared dataset files")
        X = df_train["image"]
        y = df_train["label"]
        X_train, X_test_val, y_train, y_test_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
        )
        train_indices = list(map(int, X_train.index.tolist()))
        val_indices = list(map(int, X_val.index.tolist()))
        test_indices = list(map(int, X_test.index.tolist()))

    ds_splits = DatasetDict(
        {
            "train": ds["train"].select(train_indices),
            "valid": ds["train"].select(val_indices),
            "test": ds["train"].select(test_indices),
        }
    )

    with stats_path.open() as f:
        combined_stats = json.load(f)

    logger.info("Creating label maps and computing statistics")
    id2label, label2id = build_label_maps(logger, ds, combined_stats)

    mean, std = compute_mean_std(ds_splits["train"])

    logger.info(f"Number of training samples: {len(ds_splits['train'])}")
    logger.info(f"Number of validation samples: {len(ds_splits['valid'])}")
    logger.info(f"Number of test samples: {len(ds_splits['test'])}")
    logger.info(f"Mean: {mean}")
    logger.info(f"Std: {std}")

    return ds_splits, id2label, label2id, mean, std

