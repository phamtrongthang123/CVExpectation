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
Simple object detection using Faster R-CNN.
Demonstrates: loading detection models, bounding box predictions, and class labels.
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

# Load pretrained Faster R-CNN model
print("Loading Faster R-CNN model...")
model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Preprocess image
transform = transforms.ToTensor()
input_tensor = transform(image)

# Perform object detection
with torch.no_grad():
    prediction = model([input_tensor])

# COCO class labels (Faster R-CNN is trained on COCO dataset)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Filter predictions with confidence > 0.5
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

threshold = 0.5
print(f"\nDetected objects (confidence > {threshold}):")
print("-" * 60)

for box, label, score in zip(boxes, labels, scores):
    if score > threshold:
        class_name = COCO_CLASSES[label]
        x1, y1, x2, y2 = box.tolist()
        print(f"{class_name:20s} | Confidence: {score:.3f} | Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

print(f"\nTotal detections: {(scores > threshold).sum().item()}")
