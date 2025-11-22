#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple MLOps workflow demonstrating the ML lifecycle.
Demonstrates: data versioning, model training, evaluation, saving, and deployment prep.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Configuration
class Config:
    data_version = "v1.0"
    model_version = "v1.0"
    input_dim = 10
    hidden_dim = 64
    output_dim = 3
    learning_rate = 0.001
    epochs = 50
    batch_size = 32

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

# Step 1: Data Generation and Versioning
def generate_data(n_samples=1000):
    """Generate and version data"""
    print("[1] Data Generation and Versioning")
    torch.manual_seed(42)

    X = torch.randn(n_samples, Config.input_dim)
    y = torch.randint(0, Config.output_dim, (n_samples,))

    # Create data hash for versioning
    data_hash = hashlib.md5(X.numpy().tobytes()).hexdigest()[:8]

    print(f"    Generated {n_samples} samples")
    print(f"    Data version: {Config.data_version}")
    print(f"    Data hash: {data_hash}")

    return X, y, data_hash

# Step 2: Model Training
def train_model(X, y):
    """Train the model"""
    print("\n[2] Model Training")

    model = SimpleModel(Config.input_dim, Config.hidden_dim, Config.output_dim)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model.train()
    for epoch in range(Config.epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_acc = (val_outputs.argmax(1) == y_val).float().mean()
            model.train()

            print(f"    Epoch {epoch:2d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model

# Step 3: Model Evaluation
def evaluate_model(model, X, y):
    """Evaluate model performance"""
    print("\n[3] Model Evaluation")

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(1)
        accuracy = (predictions == y).float().mean()

        # Per-class accuracy
        for cls in range(Config.output_dim):
            mask = y == cls
            if mask.sum() > 0:
                cls_acc = (predictions[mask] == y[mask]).float().mean()
                print(f"    Class {cls} accuracy: {cls_acc:.4f}")

        print(f"    Overall accuracy: {accuracy:.4f}")

    return accuracy.item()

# Step 4: Model Artifacts and Metadata
def save_model_artifacts(model, accuracy, data_hash):
    """Save model and metadata"""
    print("\n[4] Saving Model Artifacts")

    # Create artifacts directory
    artifacts_dir = Path("model_artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Save model checkpoint
    model_path = artifacts_dir / f"model_{Config.model_version}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"    Model saved to: {model_path}")

    # Save metadata
    metadata = {
        'model_version': Config.model_version,
        'data_version': Config.data_version,
        'data_hash': data_hash,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'input_dim': Config.input_dim,
            'hidden_dim': Config.hidden_dim,
            'output_dim': Config.output_dim,
            'learning_rate': Config.learning_rate,
            'epochs': Config.epochs,
        },
        'performance': {
            'accuracy': accuracy
        }
    }

    metadata_path = artifacts_dir / f"metadata_{Config.model_version}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Metadata saved to: {metadata_path}")

    return model_path, metadata_path

# Step 5: Model Deployment Preparation
def prepare_deployment(model_path):
    """Prepare model for deployment"""
    print("\n[5] Deployment Preparation")

    # Load model
    model = SimpleModel(Config.input_dim, Config.hidden_dim, Config.output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert to TorchScript for production
    script_path = model_path.with_suffix('.scripted.pt')
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(script_path))
    print(f"    TorchScript model saved: {script_path}")

    # Test inference
    with torch.no_grad():
        sample_input = torch.randn(1, Config.input_dim)
        output = scripted_model(sample_input)
        print(f"    Test inference successful: {output.argmax().item()}")

    print(f"    Model ready for deployment!")

# Main MLOps Pipeline
def main():
    print("=" * 60)
    print("Simple MLOps Pipeline")
    print("=" * 60)

    # Execute pipeline
    X, y, data_hash = generate_data()
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    model_path, metadata_path = save_model_artifacts(model, accuracy, data_hash)
    prepare_deployment(model_path)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
