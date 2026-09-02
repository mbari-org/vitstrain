# vitstrain
# Filename: src/fine_tune_vits.py
# Description: Fine-tuning a Vision Transformer model

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, AutoImageProcessor
from transformers import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import confusion_matrix
from args import parse_args
from data_utils import collate_fn, create_dataset, load_prepared_dataset
from model_factory import create_model, export_onnx
from plot_utils import plot_multiclass_pr_curves
from version import __version__
import matplotlib.pyplot as plt


BASE_BATCH_SIZE = 32

# Configure logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

def compute_per_class_metrics(y_true, y_pred, y_prob, class_names, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Computes per-class evaluation metrics and optimal thresholds.

    Combines:
    - standard multiclass classification metrics
    - one-vs-rest threshold optimization metrics

    Args:
        y_true(numpy array): True labels.
        y_pred(numpy array): Predicted labels.
        y_prob(numpy array): Predicted probabilities for each class.

    Returns:
        list[dict]: Per-class evaluation metrics.
    """

    # Standard Multiclass Metrics. Classes absent from both y_true and y_pred still get a row.
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = []

    for class_idx, class_name in enumerate(class_names):
        class_metrics = report[class_name]

        # Standard Multiclass Metrics.
        precision = float(class_metrics["precision"])
        recall    = float(class_metrics["recall"])
        f1        = float(class_metrics["f1-score"])
        support   = int(class_metrics["support"])

        # Optimal Threshold Metrics (one-vs-rest / one-vs-all).
        y_true_binary = (y_true == class_idx).astype(int)
        y_prob_binary = y_prob[:, class_idx]

        best_threshold = 0.5
        best_threshold_f1 = 0.0

        for threshold in thresholds:
            y_pred_binary = (y_prob_binary >= threshold).astype(int)

            threshold_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            if threshold_f1 > best_threshold_f1:
                best_threshold_f1 = threshold_f1
                best_threshold = threshold

        metrics = {
            "class_name":         class_name,
            "class_id":           class_idx,

            # standard multiclass metrics
            "precision":          precision,
            "recall":             recall,
            "f1_score":           f1,
            "support":            support,

            # threshold optimized metrics
            "optimal_threshold": float(best_threshold),
            "threshold_f1":      float(best_threshold_f1),
        }

        per_class_metrics.append(metrics)

    return per_class_metrics


def find_optimal_thresholds(y_true, y_prob, class_names, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Find optimal threshold for each class using F1 maximization.

    For multi-class problems, treats each class as a one-vs-all binary classification
    and finds the threshold that maximizes F1 score for that class.

    Args:
        y_true: True labels (numpy array)
        y_prob: Predicted probabilities for each class (numpy array, shape: n_samples x n_classes)
        class_names: List of class names
        thresholds: Array of threshold values to test

    Returns:
        List of dictionaries with 'class_name', 'class_id', 'threshold', 'f1_score', 'support' for each class
    """
    optimal_thresholds = []

    for class_idx, class_name in enumerate(class_names):
        # Create binary labels for one-vs-all classification
        y_true_binary = (y_true == class_idx).astype(int)
        y_prob_binary = y_prob[:, class_idx]

        # Calculate support (number of true instances of this class)
        support = int(np.sum(y_true_binary))

        best_f1 = 0
        best_threshold = 0.5  # default threshold

        # Try different thresholds
        for threshold in thresholds:
            y_pred_binary = (y_prob_binary >= threshold).astype(int)

            # Calculate F1 score
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds.append({
            'class_name': class_name,
            'class_id': class_idx,
            'threshold': best_threshold,
            'f1_score': best_f1,
            'support': support
        })

        logger.info(f"Class '{class_name}' (ID: {class_idx}): Optimal threshold = {best_threshold:.3f}, F1 = {best_f1:.3f}, Support = {support}")

    return optimal_thresholds


def get_image_size(processor):
    """Returns the image size from the processor configuration. Falls back to 224."""

    if getattr(processor, "crop_size") is not None:
        crop_size = processor.crop_size
        if isinstance(crop_size, dict):
            return crop_size["height"]
        return crop_size

    if getattr(processor, "size") is not None:
        size = processor.size
        if isinstance(size, dict):
            if "height" in size:
                return size["height"]
            if "shortest_edge" in size:
                return size["shortest_edge"]
        return size

    logger.error("No crop size found in processor. Using default size of 224.")

    return 224


def scale_lr(
    lr: float,
    batch_size: int,
    base_batch_size: int = 32,
    mode: Literal["none", "linear", "sqrt"] = "none"
) -> float:
    """Scale learning rate with the effective `batch_size`."""

    if mode == "none":
        return lr

    scale = batch_size / base_batch_size

    if mode == "linear":
        return lr * scale

    if mode == "sqrt":
        return lr * (scale ** 0.5)

    raise ValueError(f"Unknown LR scaling mode: {mode}")


# Main function
def main():
    args = parse_args()

    # Command-line argument values
    train_only = getattr(args, "train_only", False)
    remove_long_tail = args.remove_long_tail
    add_rotations = args.add_rotations
    model_name = args.model_name
    base_model = args.base_model
    remap = args.remap
    raw_data = [Path(path) for path in args.raw_data]
    exclude_labels = args.exclude_labels if args.exclude_labels else []
    filter_data = Path(args.filter_data)
    num_epochs = args.num_epochs
    early_stopping_epochs = args.early_stopping_epochs
    min_images_per_class = args.min_images_per_class
    export_to_onnx = args.export_onnx
    per_device_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    # todo: add freeze_backbone as cli flag (default currently False)

    num_devices = int(os.environ.get("WORLD_SIZE", "1"))

    effective_bs = (
        per_device_batch_size
        * num_devices
        * gradient_accumulation_steps
    )
    scaled_lr = scale_lr(
        lr=args.learning_rate,
        batch_size=effective_bs,
        base_batch_size=BASE_BATCH_SIZE,
        mode=args.lr_scaling,
    )
    auto_find_batch_size = args.lr_scaling == "none"  # avoid resize after scaling

    # Append timestamp to the model name
    now = datetime.now()
    model_name = f"{model_name}-{now:%Y%m%d}"

    # Persist the dataset split using the same dated naming convention as the model
    split_json_path = filter_data / "split.json"

    # Define loss history file
    loss_history_file = f"loss_history_{model_name}.json"

    # Log configuration
    logger.info(f"=========================vitstrain v{__version__}========================================")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Remove long-tail classes: {remove_long_tail}")
    logger.info(f"Minimum images per class: {min_images_per_class}")
    logger.info(f"Add rotations: {add_rotations}")
    logger.info(f"Early stopping epochs: {early_stopping_epochs}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Raw data paths: {[p.as_posix() for p in raw_data]}")
    logger.info(f"Excluded labels: {exclude_labels}")
    logger.info(f"Filtered data path: {filter_data}")
    logger.info(f"Split JSON path: {split_json_path}")
    logger.info(f"Remap classes: {remap}")
    logger.info(f"Train only (skip data prep): {train_only}")
    logger.info(f"Base LR: {args.learning_rate}")
    logger.info(f"LR Scaling: {args.lr_scaling}")
    logger.info(f"Base BS: {BASE_BATCH_SIZE}")
    logger.info(f"Per-device BS: {per_device_batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Number of devices: {num_devices}")
    logger.info(f"Effective BS: {effective_bs}")
    logger.info(f"Scaled LR: {scaled_lr}")
    logger.info(f"Export to ONNX: {export_to_onnx}")
    logger.info(f"Loss history file: {loss_history_file}")
    logger.info("==========================================================================")
    logger.info(f"Remove the loss history file and filtered data path if you want to restart training, e.g. rm {loss_history_file} && rm -rf {filter_data}")
    logger.info("Otherwise, the training will resume from the last checkpoint.")
    logger.info("==========================================================================")

    if remap:
        with open(remap) as f:
            remap = json.load(f)

    if train_only:
        # Load the already-prepared dataset in filter_data and reuse split.json if possible.
        ds_splits, id2label, label2id, image_mean, image_std = load_prepared_dataset(
            logger,
            filter_data,
            split_json_path=split_json_path,
        )
    else:
        # Create the dataset from the raw dataset(s)
        ds_splits, id2label, label2id, image_mean, image_std = create_dataset(
            logger,
            remove_long_tail,
            raw_data,
            filter_data,
            remap,
            exclude_labels,
            min_images_per_class,
            split_json_path=split_json_path,
        )

    train_ds = ds_splits['train']
    val_ds   = ds_splits['valid']
    test_ds  = ds_splits['test']

    # Create Model.
    model = create_model(logger, base_model, id2label)

    # Configure Preprocessing.
    processor = AutoImageProcessor.from_pretrained(base_model, use_fast=True)
    processor.image_mean = image_mean
    processor.image_std = image_std

    size = get_image_size(processor)

    _train_transforms = A.Compose([  # todo: p=1 for rotations --> 90 + 180 + 270 == 180 ?
        A.RandomResizedCrop(height=size, width=size, scale=(0.2, 1.0), p=1.0),
        *([A.Rotate(limit=90,  interpolation=1, border_mode=4, value=None, p=1)] if add_rotations else []),
        *([A.Rotate(limit=180, interpolation=1, border_mode=4, value=None, p=1)] if add_rotations else []),
        *([A.Rotate(limit=270, interpolation=1, border_mode=4, value=None, p=1)] if add_rotations else []),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.1, p=0.5),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2(),
    ])

    _val_transforms = A.Compose([  # todo: fix randomness in validation ?
        A.RandomResizedCrop(height=size, width=size, scale=(0.2, 1.0), p=1.0),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2(),
    ])

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image=np.array(i))["image"] for i in examples["image"]]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(image=np.array(i))["image"] for i in examples["image"]]
        return examples

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    # Train Model.

    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.focal_loss = FocalLoss()

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.focal_loss(logits, labels)

            return (loss, outputs) if return_outputs else loss

    class LossLoggerCallback(TrainerCallback):
        def __init__(self, save_path="loss_history.json"):
            self.loss_history = {"train_loss": [], "eval_loss": []}
            self.save_path = save_path
            self._load_history()

        def _load_history(self):
            """Load existing loss history from a file if it exists."""
            try:
                with open(self.save_path, "r") as f:
                    self.loss_history = json.load(f)
            except FileNotFoundError:
                self.loss_history = {"train_loss": [], "eval_loss": []}
            except json.JSONDecodeError:
                logger.warning(f"Error loading loss history from {self.save_path}")
                self.loss_history = {"train_loss": [], "eval_loss": []}

        def _save_history(self):
            """Save the current loss history to a file."""
            with open(self.save_path, "w") as f:
                json.dump(self.loss_history, f)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                if "loss" in logs:
                    self.loss_history["train_loss"].append(logs["loss"])
                if "eval_loss" in logs:
                    self.loss_history["eval_loss"].append(logs["eval_loss"])
                self._save_history()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    loss_logger = LossLoggerCallback(save_path=loss_history_file)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_epochs)

    train_args = TrainingArguments(
        model_name,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=scaled_lr,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_batch_size,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        logging_steps=10,  # Log every 10 steps
        remove_unused_columns=False,
        auto_find_batch_size=auto_find_batch_size,
    )

    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[loss_logger, early_stopping],
    )

    checkpoint = get_last_checkpoint(model_name)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(model_name)

    if export_to_onnx:
        export_onnx(logger, trainer.model, Path(model_name) / "model.onnx", size)

    # Run predictions on the test and val datasets
    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    outputs = trainer.predict(test_ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    y_prob = torch.nn.functional.softmax(torch.tensor(outputs.predictions), dim=-1).numpy()

    # Compute Global Metrics.
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    # Compute Per-Class Metrics.
    class_names = list(id2label.values())
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, y_prob, class_names)
    optimal_thresholds = find_optimal_thresholds(y_true, y_prob, class_names)  # todo: redundant

    # Log Metrics.
    logger.info(
        f"Accuracy: {accuracy:.2f}, "
        f"Precision: {precision:.2f}, "
        f"Recall: {recall:.2f}"
    )
    for m in per_class_metrics:
        logger.info(
            f"Class \'{m['class_name']}\' (ID: {m['class_id']}): "
            f"Precision={m['precision']:.3f}, "
            f"Recall={m['recall']:.3f}, "
            f"F1={m['f1_score']:.3f}, "
            f"Support={m['support']}, "
            f"Optimal Threshold={m['optimal_threshold']:.3f}, "
            f"Threshold F1={m['threshold_f1']:.3f}"
        )

    # Save Metrics.
    pcm_path = Path(model_name) / "per_class_metrics.csv"
    with open(pcm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_metrics[0]))
        writer.writeheader()
        writer.writerows(per_class_metrics)
    logger.info(f"Per-class metrics saved to {pcm_path}")

    csv_filename = Path(model_name) / f"optimal_thresholds_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['class_name', 'class_id', 'f1_score', 'support', 'threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(optimal_thresholds)
    logger.info(f"Optimal thresholds saved to {csv_filename}")

    all_labels = id2label.values()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_labels)))

    # Normalize the confusion matrix to range 0-1, leaving rows with no test samples at 0
    row_totals = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype('float'), row_totals, out=np.zeros(cm.shape), where=row_totals != 0)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_normalized, xticklabels=all_labels, yticklabels=all_labels, cmap='Blues')

    # Generate precision-recall plot by class
    class_names = list(id2label.values())
    pr_curve_path = plot_multiclass_pr_curves(y_true, y_prob, class_names, model_name)
    logger.info(f"Precision-recall curves saved to {pr_curve_path.name}")

    # Plot confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.suptitle(
        f"CM {model_name}. Top-1 Balanced Accuracy: {accuracy:.2f},  "
        f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    d = f"{datetime.now():%Y-%m-%d_%H%M%S}"
    plt.title(d)
    plot_path = Path(model_name) / f"confusion_matrix_{model_name}_{d}.png"
    logger.info(f"Saving confusion matrix to {plot_path.name}")
    plt.savefig(plot_path.as_posix())
    plt.close()

    # Plot the loss curves if there are at least a few points
    if len(loss_logger.loss_history["train_loss"]) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_logger.loss_history["train_loss"], label="Training Loss", color="blue")
        eval_steps = list(range(0, len(loss_logger.loss_history["train_loss"]),
                                len(loss_logger.loss_history["train_loss"]) // len(loss_logger.loss_history["eval_loss"])))
        eval_steps = eval_steps[:len(loss_logger.loss_history["eval_loss"])]
        plt.plot(eval_steps, loss_logger.loss_history["eval_loss"], label="Validation Loss", color="orange")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {model_name}")
        plt.legend()
        loss_curve_path = Path(model_name) / f"loss_curve_{model_name}_{datetime.now():%Y-%m-%d_%H%M%S}.png"
        plt.savefig(loss_curve_path.as_posix())
        logger.info(f"Loss curve saved to {loss_curve_path.name}")
        plt.close()

    # TODO: add version to something for provenance

    # Push to the HuggingFace model hub
    # trainer.push_to_hub()


if __name__ == "__main__":
    main()
