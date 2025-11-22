#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
#     "matplotlib",
# ]
# ///

"""
Simple comparative study between different optimization algorithms.
Demonstrates: comparing SGD, Adam, and RMSprop on the same task.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("Comparative Study: Optimization Algorithms")
print("=" * 70)

# ============================================================================
# 1. Setup: Define model and data
# ============================================================================
print("\n[1] Setting up experiment")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.layers(x)

# Generate synthetic dataset
torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 10)
y = torch.randint(0, 3, (n_samples,))

# Split into train/val
split = int(0.8 * n_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"   Dataset: {n_samples} samples, 10 features, 3 classes")
print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

# ============================================================================
# 2. Define optimizers to compare
# ============================================================================
print("\n[2] Optimizers to compare:")

optimizers_config = [
    {'name': 'SGD', 'class': torch.optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
    {'name': 'Adam', 'class': torch.optim.Adam, 'params': {'lr': 0.001}},
    {'name': 'RMSprop', 'class': torch.optim.RMSprop, 'params': {'lr': 0.001}},
    {'name': 'AdaGrad', 'class': torch.optim.Adagrad, 'params': {'lr': 0.01}},
]

for config in optimizers_config:
    print(f"   - {config['name']}: {config['params']}")

# ============================================================================
# 3. Train models with different optimizers
# ============================================================================
print("\n[3] Training models...")

epochs = 100
results = {}

for config in optimizers_config:
    print(f"\n   Training with {config['name']}...")

    # Create fresh model
    model = SimpleModel()
    optimizer = config['class'](model.parameters(), **config['params'])
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_acc = (val_outputs.argmax(1) == y_val).float().mean()
            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())

        if epoch % 20 == 0:
            print(f"      Epoch {epoch:3d}: train_loss={loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    results[config['name']] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_val_acc': val_accs[-1],
        'final_val_loss': val_losses[-1]
    }

    print(f"   ✓ Final accuracy: {val_accs[-1]:.4f}")

# ============================================================================
# 4. Compare results
# ============================================================================
print("\n[4] Comparison Results")
print("   " + "-" * 66)
print(f"   {'Optimizer':<15} {'Final Val Loss':>15} {'Final Val Acc':>15} {'Best Acc':>15}")
print("   " + "-" * 66)

for name, result in results.items():
    best_acc = max(result['val_accs'])
    print(f"   {name:<15} {result['final_val_loss']:>15.4f} {result['final_val_acc']:>15.4f} {best_acc:>15.4f}")

print("   " + "-" * 66)

# Find best optimizer
best_optimizer = max(results.items(), key=lambda x: x[1]['final_val_acc'])
print(f"\n   🏆 Best performer: {best_optimizer[0]} (Val Acc: {best_optimizer[1]['final_val_acc']:.4f})")

# ============================================================================
# 5. Visualization
# ============================================================================
print("\n[5] Creating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparative Study: Optimization Algorithms', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
for name, result in results.items():
    axes[0, 0].plot(result['train_losses'], label=name, linewidth=2)
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Validation Loss
for name, result in results.items():
    axes[0, 1].plot(result['val_losses'], label=name, linewidth=2)
axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Validation Accuracy
for name, result in results.items():
    axes[1, 0].plot(result['val_accs'], label=name, linewidth=2)
axes[1, 0].set_title('Validation Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final Performance Comparison
names = list(results.keys())
final_accs = [results[name]['final_val_acc'] for name in names]
colors = ['green' if name == best_optimizer[0] else 'steelblue' for name in names]

axes[1, 1].bar(names, final_accs, color=colors, edgecolor='black')
axes[1, 1].set_title('Final Validation Accuracy Comparison')
axes[1, 1].set_xlabel('Optimizer')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim([min(final_accs) * 0.95, max(final_accs) * 1.05])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (name, acc) in enumerate(zip(names, final_accs)):
    axes[1, 1].text(i, acc, f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved visualization: optimizer_comparison.png")

print("\n" + "=" * 70)
print("Comparative study completed!")
