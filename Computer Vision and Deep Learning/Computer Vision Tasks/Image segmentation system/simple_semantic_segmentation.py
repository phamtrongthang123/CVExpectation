#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow",
#     "numpy",
# ]
# ///

"""
Simple semantic segmentation using DeepLabV3.
Demonstrates: loading segmentation models and pixel-wise classification.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Create a synthetic image for demonstration
print("Creating synthetic image...")
synthetic_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
image = Image.fromarray(synthetic_image, mode='RGB')

# Load pretrained DeepLabV3 model
print("Loading DeepLabV3 model...")
model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess image
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Perform segmentation
with torch.no_grad():
    output = model(input_batch)['out'][0]

# Get predicted class for each pixel
output_predictions = output.argmax(0)

# Create a color palette for visualization (21 classes in PASCAL VOC)
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Map predictions to colors
segmentation_mask = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image.size)
segmentation_mask.putpalette(colors.flatten().tolist())

print("Segmentation completed!")
print(f"Found {len(torch.unique(output_predictions))} unique classes in the image")
print(f"Image size: {image.size}")
print(f"Segmentation mask created - would be saved as 'segmentation_result.png'")

# In a real application, you would save the result:
# segmentation_mask.save("segmentation_result.png")
