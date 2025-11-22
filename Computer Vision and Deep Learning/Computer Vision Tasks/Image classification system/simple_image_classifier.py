#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "torchvision",
# ]
# ///

"""
Simple image classification using a pretrained ResNet model.
Demonstrates: loading pretrained models, image preprocessing, and making predictions.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Create a synthetic image (instead of downloading)
print("Creating synthetic image for demonstration...")
synthetic_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(synthetic_image)
print("Synthetic image created")

# Load pretrained ResNet18 model
print("Loading ResNet18 model...")
model = models.resnet18(weights='DEFAULT')
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess image
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(input_batch)

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Use simplified class labels (ImageNet has 1000 classes)
print("\nTop 5 predictions:")
for i in range(5):
    print(f"{i+1}. Class {top5_catid[i].item()}: {top5_prob[i].item()*100:.2f}%")
