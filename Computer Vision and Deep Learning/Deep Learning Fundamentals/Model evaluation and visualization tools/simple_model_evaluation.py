#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple model evaluation and visualization tools.
Demonstrates: metrics calculation, confusion matrix, and performance analysis.
"""

import torch
import numpy as np

# Simulate predictions and ground truth
num_samples = 100
num_classes = 3

# Generate synthetic predictions and labels
torch.manual_seed(42)
predictions = torch.randn(num_samples, num_classes)
pred_labels = predictions.argmax(dim=1)
true_labels = torch.randint(0, num_classes, (num_samples,))

print("Model Evaluation Metrics")
print("=" * 60)

# 1. Accuracy
accuracy = (pred_labels == true_labels).float().mean()
print(f"\nAccuracy: {accuracy.item():.4f} ({accuracy.item()*100:.2f}%)")

# 2. Per-class accuracy
for cls in range(num_classes):
    mask = true_labels == cls
    if mask.sum() > 0:
        cls_acc = (pred_labels[mask] == true_labels[mask]).float().mean()
        print(f"  Class {cls} accuracy: {cls_acc.item():.4f} ({mask.sum()} samples)")

# 3. Confusion Matrix
confusion_matrix = torch.zeros(num_classes, num_classes)
for t, p in zip(true_labels, pred_labels):
    confusion_matrix[t, p] += 1

print(f"\nConfusion Matrix:")
print("       Predicted:")
print("       ", end="")
for i in range(num_classes):
    print(f"  {i:3d}", end="")
print()

for i in range(num_classes):
    print(f"True {i}:", end="")
    for j in range(num_classes):
        print(f"{int(confusion_matrix[i, j]):4d}", end="")
    print()

# 4. Precision, Recall, F1 for each class
print(f"\nPer-class Metrics:")
print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 42)

for cls in range(num_classes):
    tp = confusion_matrix[cls, cls]
    fp = confusion_matrix[:, cls].sum() - tp
    fn = confusion_matrix[cls, :].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{cls:6d} {precision:10.4f} {recall:10.4f} {f1:10.4f}")

# 5. Loss calculation example
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(predictions, true_labels)
print(f"\nCross-Entropy Loss: {loss.item():.4f}")

# 6. Confidence statistics
probs = torch.softmax(predictions, dim=1)
max_probs, _ = probs.max(dim=1)
print(f"\nPrediction Confidence:")
print(f"  Mean: {max_probs.mean():.4f}")
print(f"  Std:  {max_probs.std():.4f}")
print(f"  Min:  {max_probs.min():.4f}")
print(f"  Max:  {max_probs.max():.4f}")

# 7. Top-k accuracy
k = 2
top_k_preds = predictions.topk(k, dim=1)[1]
top_k_acc = (top_k_preds == true_labels.unsqueeze(1)).any(dim=1).float().mean()
print(f"\nTop-{k} Accuracy: {top_k_acc.item():.4f}")
