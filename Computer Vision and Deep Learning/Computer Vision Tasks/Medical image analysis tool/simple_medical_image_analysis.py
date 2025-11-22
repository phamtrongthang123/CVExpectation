#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
#     "pillow",
# ]
# ///

"""
Simple medical image analysis demonstrating basic preprocessing and feature extraction.
Demonstrates: image normalization, contrast enhancement, and simple tumor detection simulation.
"""

import torch
import numpy as np
from PIL import Image, ImageEnhance
import torch.nn as nn

# Simulate a medical image (grayscale CT scan)
# In reality, you'd load DICOM files using libraries like pydicom
def create_synthetic_ct_scan(size=256):
    """Create a synthetic CT scan with a simulated anomaly"""
    # Background tissue
    image = np.random.normal(100, 20, (size, size))

    # Add a circular anomaly (simulated tumor)
    center = (size // 2, size // 2)
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 20**2
    image[mask] = np.random.normal(180, 10, mask.sum())

    # Clip to valid HU range (simplified)
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# Create synthetic medical image
medical_image = create_synthetic_ct_scan(256)
print(f"Created synthetic CT scan: {medical_image.shape}")

# Convert to PIL and apply preprocessing
img_pil = Image.fromarray(medical_image, mode='L')

# Apply contrast enhancement (common in medical imaging)
enhancer = ImageEnhance.Contrast(img_pil)
enhanced_img = enhancer.enhance(1.5)

# Convert to tensor for processing
img_tensor = torch.from_numpy(np.array(enhanced_img)).float()

# Normalize to [0, 1]
img_normalized = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

# Simple thresholding to detect bright regions (potential anomalies)
threshold = img_normalized.mean() + 2 * img_normalized.std()
anomaly_mask = img_normalized > threshold

# Calculate statistics
print(f"\nImage statistics:")
print(f"  Mean intensity: {img_normalized.mean():.2f}")
print(f"  Std intensity: {img_normalized.std():.2f}")
print(f"  Detected anomaly pixels: {anomaly_mask.sum().item()}")
print(f"  Anomaly percentage: {(anomaly_mask.sum() / anomaly_mask.numel() * 100):.2f}%")

# Simple CNN for feature extraction (not trained, just architecture demo)
class MedicalImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 64 * 64, 2)  # Binary: normal vs abnormal

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = MedicalImageCNN()
print(f"\nModel architecture for medical image classification created")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
