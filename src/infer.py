# vittrain
# Filename: src/infer.py
# Description: Simple example on using a fine-tuned a Vision Transformer model with a HuggingFace Model

from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

model_name='mbari-uav-vit-b-16'
model = AutoModelForImageClassification.from_pretrained(model_name)
model.to("cuda")
processor = AutoImageProcessor.from_pretrained(model_name)
image_paths = ["/tmp/plane_cifar10.png"]
image_paths = ["/tmp/UAV/Baseline/crops/Mola/2461655.jpg",
               "/tmp/UAV/Baseline/crops/Pinniped/2494504.jpg",
               "/tmp/UAV/Baseline/crops/Shark/1991348.jpg",
               "/tmp/UAV/Baseline/crops/Surfboard/1991103view_r.jpg"]
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
inputs = processor(images=images, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).cpu().numpy()
    for i, predicted_class_idx in enumerate(predicted_class_idx):
        print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")