#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow",
# ]
# ///

"""
Simple data augmentation pipeline using torchvision transforms.
Demonstrates: various augmentation techniques for training robustness.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Create a synthetic image for demonstration
print("Creating synthetic image...")
synthetic_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
image = Image.fromarray(synthetic_image, mode='RGB')

print(f"Original image size: {image.size}")

# Define augmentation pipeline
augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply augmentation multiple times to show variation
print("\nApplying augmentation pipeline 5 times:")
for i in range(5):
    augmented = augmentation_pipeline(image)
    print(f"  Augmentation {i+1}: shape={augmented.shape}, "
          f"mean={augmented.mean():.3f}, std={augmented.std():.3f}")

# Advanced augmentation: Cutout (random erasing)
advanced_augmentation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

print("\nAdvanced augmentation with random erasing:")
augmented = advanced_augmentation(image)
print(f"  Shape: {augmented.shape}")

# Custom augmentation function
def custom_augment(img, noise_level=0.1):
    """Add Gaussian noise to image"""
    img_tensor = transforms.ToTensor()(img)
    noise = torch.randn_like(img_tensor) * noise_level
    noisy_img = torch.clamp(img_tensor + noise, 0, 1)
    return noisy_img

print("\nCustom noise augmentation:")
noisy = custom_augment(image, noise_level=0.05)
print(f"  Noisy image shape: {noisy.shape}, mean: {noisy.mean():.3f}")

# Create a training-ready augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Validation transform (no augmentation, just normalization)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\nTraining and validation transforms created successfully!")
print("Use train_transform for training data and val_transform for validation/test data")
