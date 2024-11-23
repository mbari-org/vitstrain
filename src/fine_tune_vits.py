# vittrain
# Filename: src/fine_tune_vits.py
# Description: Fine-tuning a Vision Transformer model with HuggingFace

import logging
from datetime import datetime
from pathlib import Path
from albumentations import (
    Compose,
    Resize,
    RandomResizedCrop,
    HorizontalFlip,
    Normalize,
    CenterCrop,
    Rotate
)
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score
import seaborn as sns
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import collate_fn, create_dataset
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# Log to the console
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

# Name of the model you want to train
model_name = 'mbari-uav-vit-b-16'

# The raw dataset and the place to store the filtered dataset
raw_data = Path('/tmp/UAV/Baseline/')
filter_data = Path('/tmp/UAV/Baseline_filter/')
# raw_data = Path('/tmp/catsdogs/catsdogstrain')
# filter_data = Path('/tmp/catsdogs/catsdogstrain')

# Create the dataset from the raw dataset(s)
ds_splits, id2label, label2id = create_dataset(logger, [raw_data], filter_data)

# The id2label and label2id are used to convert the labels to and from the model's internal representation
# These are stored in the HuggingFace config.json file with the model, e.g. mbari-uav-vit-b-16/config.json
base_model = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(base_model,
                                                  id2label=id2label,
                                                  label2id=label2id)

train_ds = ds_splits['train']
val_ds = ds_splits['valid']
test_ds = ds_splits['test']

# Image processor and transforms. The transforms may be replaced with albumentations
processor = ViTImageProcessor.from_pretrained(base_model)

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Mean and standard deviation for normalization
image_mean = [0.485, 0.456, 0.406]  # Example values
image_std = [0.229, 0.224, 0.225]   # Example values
size = 224  # Example size

# Training transforms
_train_transforms = Compose(
    [
        RandomResizedCrop(height=size, width=size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        HorizontalFlip(p=0.5),
        Rotate(limit=[0, 360], step=45, p=1.0),  # Rotate in 45-degree increments
        Normalize(mean=image_mean, std=image_std),
        ToTensorV2(),  # Converts to PyTorch tensor
    ]
)

# Validation transforms
_val_transforms = Compose(
    [
        Resize(height=size, width=size),
        CenterCrop(height=size, width=size),
        Rotate(limit=[0, 360], step=45, p=1.0),  # Rotate in 45-degree increments
        Normalize(mean=image_mean, std=image_std),
        ToTensorV2(),  # Converts to PyTorch tensor
    ]
)

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

args = TrainingArguments(
    model_name,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=20,
    num_train_epochs=100,
    gradient_accumulation_steps=2,
    save_total_limit = 1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

# HuggingFace Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# Train the model and save it. This will save the model to a directory of the same name
# If checkpoints exist, load the best model from the checkpoint
if Path(model_name).exists() and len(list(Path(model_name).rglob('*.safetensors'))) > 0:
    trainer.train(resume_from_checkpoint = True)
else:
    trainer.train()
trainer.save_model(model_name)

# Run predictions on the test set. More work needed here to save the confusion matrix and other metrics
# This will output a confusion matrix in the blue color map with only index labels
outputs = trainer.predict(test_ds)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

accuracy = accuracy_score(y_true, y_pred)
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
            f"CM {model_name} exemplars. Top-1 Accuracy: {accuracy:.2f},  "
                f"Precision: {precision:.2f}, Recall: {recall:.2f}")
d = f"{datetime.now():%Y-%m-%d %H%M%S}"
plt.title(d)
plot_name = f"confusion_matrix_{model_name}_{d}.png"
logger.info(f"Saving confusion matrix to {plot_name}")
plt.savefig(plot_name)
plt.close()
