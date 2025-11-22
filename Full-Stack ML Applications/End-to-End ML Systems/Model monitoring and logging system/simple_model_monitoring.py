#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple model monitoring and logging system.
Demonstrates: tracking metrics, logging, and model performance monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json

# Simple logging class
class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_history = []
        self.start_time = datetime.now()

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Log metrics for an epoch"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc)
        }
        self.metrics_history.append(log_entry)

    def log_prediction(self, input_data, prediction, true_label, confidence):
        """Log individual prediction"""
        return {
            'timestamp': datetime.now().isoformat(),
            'input_shape': list(input_data.shape),
            'prediction': int(prediction),
            'true_label': int(true_label),
            'confidence': float(confidence),
            'correct': bool(prediction == true_label)
        }

    def get_summary(self):
        """Get training summary"""
        if not self.metrics_history:
            return "No metrics logged yet"

        latest = self.metrics_history[-1]
        best_val_loss = min(m['val_loss'] for m in self.metrics_history)
        best_val_acc = max(m['val_acc'] for m in self.metrics_history)

        summary = f"""
Model Monitoring Summary: {self.model_name}
{'=' * 60}
Total Epochs: {len(self.metrics_history)}
Training Duration: {datetime.now() - self.start_time}

Latest Metrics (Epoch {latest['epoch']}):
  Train Loss: {latest['train_loss']:.4f}
  Val Loss:   {latest['val_loss']:.4f}
  Train Acc:  {latest['train_acc']:.4f}
  Val Acc:    {latest['val_acc']:.4f}

Best Performance:
  Best Val Loss: {best_val_loss:.4f}
  Best Val Acc:  {best_val_acc:.4f}
"""
        return summary

    def save_logs(self, filepath='model_logs.json'):
        """Save logs to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'start_time': self.start_time.isoformat(),
                'metrics_history': self.metrics_history
            }, f, indent=2)
        print(f"Logs saved to {filepath}")

# Demo: Simple model training with monitoring
print("Model Monitoring and Logging Demo")
print("=" * 60)

# Initialize monitor
monitor = ModelMonitor("SimpleClassifier")

# Simulate training loop
print("\nSimulating training with monitoring...")
for epoch in range(10):
    # Simulate metrics (in real scenario, these come from actual training)
    train_loss = 1.0 - (epoch * 0.08) + np.random.uniform(-0.05, 0.05)
    val_loss = 1.0 - (epoch * 0.07) + np.random.uniform(-0.05, 0.05)
    train_acc = 0.5 + (epoch * 0.04) + np.random.uniform(-0.02, 0.02)
    val_acc = 0.5 + (epoch * 0.035) + np.random.uniform(-0.02, 0.02)

    monitor.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

    if epoch % 3 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# Display summary
print(monitor.get_summary())

# Simulate predictions with logging
print("\nLogging sample predictions:")
for i in range(5):
    input_data = torch.randn(1, 10)
    prediction = np.random.randint(0, 3)
    true_label = np.random.randint(0, 3)
    confidence = np.random.uniform(0.6, 0.99)

    pred_log = monitor.log_prediction(input_data, prediction, true_label, confidence)
    status = "✓" if pred_log['correct'] else "✗"
    print(f"  {status} Pred: {prediction}, True: {true_label}, Conf: {confidence:.3f}")

# Save logs
monitor.save_logs('model_logs.json')

# Alert system example
def check_performance_degradation(monitor, threshold=0.1):
    """Simple alert system for performance degradation"""
    if len(monitor.metrics_history) < 3:
        return

    recent_acc = [m['val_acc'] for m in monitor.metrics_history[-3:]]
    if recent_acc[-1] < recent_acc[0] - threshold:
        print(f"\n⚠️  ALERT: Performance degradation detected!")
        print(f"   Val accuracy dropped from {recent_acc[0]:.4f} to {recent_acc[-1]:.4f}")

check_performance_degradation(monitor)
