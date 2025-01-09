# vittrain
# Filename: src/fine_tune_vits.py
# Description: Fine-tuning a Vision Transformer model with HuggingFace

import logging
from datetime import datetime
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from transformers import  AutoModelForImageClassification,ViTForImageClassification,AutoImageProcessor,TrainerCallback,EarlyStoppingCallback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import collate_fn, create_dataset
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# Log to the console
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

# Set to true to truncate the long-tail classes
remove_long_tail = True

# Name of the model you want to train
now = datetime.now()
model_name = f'catsdogs-vit-b-16-{now:%Y%m%d}'

# Name of the loss history file which should be aligned with the model
loss_history_file = f'loss_history_{model_name}.json'

# The raw dataset and the place to store the filtered dataset
# Dataset assumed to be the base directory called "crops" with subdirectories per class
# e.g. /tmp/UAV/Baseline/crops/Class1, /tmp/UAV/Baseline/crops/Class2, etc.
# raw_data = Path('/tmp/UAV/Baseline/')
# filter_data = Path('/tmp/UAV/Baseline_filter/')
raw_data = [Path(__file__).parent.parent / 'data'] # raw_data is a list of paths
filter_data = Path(__file__).parent.parent / 'data_filter'

# Create the dataset from the raw dataset(s)
ds_splits, id2label, label2id, image_mean, image_std = create_dataset(logger, remove_long_tail, raw_data, filter_data)

# The id2label and label2id are used to convert the labels to and from the model's internal representation
# These are stored in the HuggingFace config.json file with the model, e.g. mbari-uav-vit-b-16/config.json
base_model = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(base_model,
                                                  num_labels=len(label2id.keys()),
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  ignore_mismatched_sizes=True,
                                                  )

train_ds = ds_splits['train']
val_ds = ds_splits['valid']
test_ds = ds_splits['test']

# Image processor and transforms
processor = AutoImageProcessor.from_pretrained(base_model, use_fast=True)
size = processor.size["height"]

# Training transforms
_train_transforms = A.Compose(
    [
        A.RandomResizedCrop(height=size, width=size, scale=(0.2, 1.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.1, p=0.5),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2(), 
    ]
)

# Validation transforms
_val_transforms = A.Compose(
    [
        A.RandomResizedCrop(height=size, width=size, scale=(0.2, 1.0), p=1.0),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2(), 
    ]
)

# Custom Focal Loss to handle class imbalance
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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss.forward(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_transforms(examples):
    examples["pixel_values"] = [_train_transforms(image=np.array(i))["image"] for i in examples["image"]] 
    return examples

def val_transforms(examples):
    examples["pixel_values"] = [_val_transforms(image=np.array(i))["image"] for i in examples["image"]] 
    return examples

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

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

args = TrainingArguments(
    model_name,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=4,
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    save_total_limit = 1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    logging_steps=10,  # Log every 10 steps
    remove_unused_columns=False,
    auto_find_batch_size=True,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=balanced_accuracy_score(predictions, labels))


loss_logger = LossLoggerCallback(save_path=loss_history_file)
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[loss_logger,early_stopping],
)

# Train the model and save it. This will save the model to a directory of the same name
# If checkpoints exist, load the best model from the checkpoint
if Path(model_name).exists() and len(list(Path(model_name).rglob('*.safetensors'))) > 0:
    trainer.train(resume_from_checkpoint = True)
else:
    trainer.train()
trainer.save_model(model_name)

# Run predictions on the test and val datasets
outputs = trainer.predict(test_ds)

metrics = trainer.evaluate(val_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# Output other metrics and confusion matrix 
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

accuracy = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='micro')
recall = recall_score(y_true, y_pred, average='micro')
logger.info(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

all_labels = id2label.values()
cm = confusion_matrix(y_true, y_pred, labels=range(len(all_labels)))

# Normalize the confusion matrix to range 0-1
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 12))
sns.heatmap(cm_normalized, xticklabels=all_labels, yticklabels=all_labels, cmap='Blues')

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

# Push to the HuggingFace model hub
# trainer.push_to_hub()
