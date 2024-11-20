# vittrain
# Filename: src/fine_tune_vits.py
# Description: Fine-tuning a Vision Transformer model with HuggingFace

import logging
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
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

# Create the dataset from the raw dataset
ds_splits, id2label, label2id = create_dataset(logger, raw_data, filter_data)

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

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
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
    f"{model_name}-finetuned",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=20,
    num_train_epochs=100,
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
trainer.train()
trainer.save_model(model_name)

# Run predictions on the test set. More work needed here to save the confusion matrix and other metrics
# This will output a confusion matrix in the blue color map with only index labels
outputs = trainer.predict(test_ds)

logger.info(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
