#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple real-time inference system with batching and optimization.
Demonstrates: model loading, batch inference, and performance optimization.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from collections import deque

# Simple model for inference
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Inference server class
class InferenceServer:
    def __init__(self, model, batch_size=32, max_wait_time=0.1):
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = deque()
        self.total_requests = 0
        self.total_latency = 0

    @torch.no_grad()
    def predict_single(self, input_data):
        """Single sample inference"""
        start_time = time.time()

        # Ensure input is batched
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)

        # Inference
        output = self.model(input_data)
        prediction = output.argmax(dim=1)
        confidence = torch.softmax(output, dim=1).max(dim=1)[0]

        latency = time.time() - start_time
        self.total_requests += 1
        self.total_latency += latency

        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'latency_ms': latency * 1000
        }

    @torch.no_grad()
    def predict_batch(self, batch_inputs):
        """Batch inference (more efficient)"""
        start_time = time.time()

        # Inference
        outputs = self.model(batch_inputs)
        predictions = outputs.argmax(dim=1)
        confidences = torch.softmax(outputs, dim=1).max(dim=1)[0]

        latency = time.time() - start_time
        self.total_requests += len(batch_inputs)
        self.total_latency += latency

        results = []
        for pred, conf in zip(predictions, confidences):
            results.append({
                'prediction': pred.item(),
                'confidence': conf.item(),
                'latency_ms': (latency / len(batch_inputs)) * 1000
            })

        return results

    def get_stats(self):
        """Get server statistics"""
        avg_latency = (self.total_latency / self.total_requests * 1000) if self.total_requests > 0 else 0
        throughput = self.total_requests / self.total_latency if self.total_latency > 0 else 0

        return {
            'total_requests': self.total_requests,
            'avg_latency_ms': avg_latency,
            'throughput_qps': throughput
        }

# Demo
print("Real-Time Inference System Demo")
print("=" * 60)

# Initialize model and server
model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=3)
server = InferenceServer(model, batch_size=32)

# Test single inference
print("\n[1] Single Sample Inference")
sample_input = torch.randn(10)
result = server.predict_single(sample_input)
print(f"  Prediction: {result['prediction']}")
print(f"  Confidence: {result['confidence']:.4f}")
print(f"  Latency: {result['latency_ms']:.2f} ms")

# Test batch inference
print("\n[2] Batch Inference (100 samples)")
batch_size = 100
batch_input = torch.randn(batch_size, 10)
results = server.predict_batch(batch_input)
print(f"  Processed {len(results)} samples")
print(f"  Avg latency per sample: {results[0]['latency_ms']:.2f} ms")

# Performance comparison
print("\n[3] Performance Comparison")
# Single inference
single_start = time.time()
for _ in range(100):
    _ = server.predict_single(torch.randn(10))
single_time = time.time() - single_start

# Batch inference
batch_start = time.time()
_ = server.predict_batch(torch.randn(100, 10))
batch_time = time.time() - batch_start

print(f"  100 single inferences: {single_time*1000:.2f} ms")
print(f"  1 batch inference (100): {batch_time*1000:.2f} ms")
print(f"  Speedup: {single_time/batch_time:.2f}x")

# Server statistics
print("\n[4] Server Statistics")
stats = server.get_stats()
print(f"  Total requests: {stats['total_requests']}")
print(f"  Average latency: {stats['avg_latency_ms']:.2f} ms")
print(f"  Throughput: {stats['throughput_qps']:.2f} requests/sec")

# Optimization tips
print("\n" + "=" * 60)
print("Optimization Tips:")
print("  - Use batch inference when possible (much faster)")
print("  - Convert model to TorchScript for production")
print("  - Use torch.compile() for PyTorch 2.0+")
print("  - Consider ONNX runtime for cross-platform deployment")
print("  - Use GPU for larger models and batches")
