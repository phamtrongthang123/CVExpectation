#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple novel architecture: Attention-based Feature Fusion Network.
Demonstrates: custom neural network architecture with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("Novel Architecture: Attention-based Feature Fusion Network")
print("=" * 70)

# ============================================================================
# Custom Attention Module
# ============================================================================
class AttentionModule(nn.Module):
    """Simple self-attention mechanism"""

    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = np.sqrt(attention_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)  # (batch, seq_len, attention_dim)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_values = torch.matmul(attention_weights, V)

        return attended_values, attention_weights

# ============================================================================
# Feature Fusion Module
# ============================================================================
class FeatureFusionModule(nn.Module):
    """Fuses multiple feature representations"""

    def __init__(self, feature_dims, output_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(feature_dims)) / len(feature_dims))

    def forward(self, features_list):
        # Project all features to same dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, projected))

        return fused, weights

# ============================================================================
# Complete Novel Architecture
# ============================================================================
class AttentionFeatureFusionNet(nn.Module):
    """
    Novel architecture combining:
    - Multi-stream feature extraction
    - Attention mechanism
    - Feature fusion
    - Final classification
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Multi-stream feature extractors
        self.stream1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        self.stream2 = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 64)
        )

        self.stream3 = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 64)
        )

        # Attention module (outputs attention_dim)
        self.attention = AttentionModule(input_dim=64, attention_dim=64)

        # Feature fusion (receives attention_dim from each stream)
        self.fusion = FeatureFusionModule(
            feature_dims=[64, 64, 64],  # Output dims from attention
            output_dim=128
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Extract features from multiple streams
        feat1 = self.stream1(x)
        feat2 = self.stream2(x)
        feat3 = self.stream3(x)

        # Apply attention to each feature stream
        # Reshape for attention (add sequence dimension)
        feat1_seq = feat1.unsqueeze(1)
        feat2_seq = feat2.unsqueeze(1)
        feat3_seq = feat3.unsqueeze(1)

        attended1, attn_weights1 = self.attention(feat1_seq)
        attended2, attn_weights2 = self.attention(feat2_seq)
        attended3, attn_weights3 = self.attention(feat3_seq)

        # Remove sequence dimension
        attended1 = attended1.squeeze(1)
        attended2 = attended2.squeeze(1)
        attended3 = attended3.squeeze(1)

        # Fuse features
        fused_features, fusion_weights = self.fusion([attended1, attended2, attended3])

        # Classification
        output = self.classifier(fused_features)

        return output, {
            'fusion_weights': fusion_weights,
            'attention_weights': [attn_weights1, attn_weights2, attn_weights3]
        }

# ============================================================================
# Demo: Train and evaluate the architecture
# ============================================================================
print("\n[1] Model Architecture")

model = AttentionFeatureFusionNet(input_dim=20, num_classes=5)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"\n   Architecture components:")
print(f"   - 3 parallel feature extraction streams")
print(f"   - Attention mechanism for each stream")
print(f"   - Learnable feature fusion")
print(f"   - Final classification layer")

# ============================================================================
# Generate synthetic data and train
# ============================================================================
print("\n[2] Training on Synthetic Data")

torch.manual_seed(42)
n_samples = 1000
X = torch.randn(n_samples, 20)
y = torch.randint(0, 5, (n_samples,))

# Split data
split = int(0.8 * n_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 50
print(f"\n   Training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs, info = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs, val_info = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_acc = (val_outputs.argmax(1) == y_val).float().mean()

        print(f"   Epoch {epoch:2d}: train_loss={loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# ============================================================================
# Analyze learned components
# ============================================================================
print("\n[3] Analyzing Learned Components")

model.eval()
with torch.no_grad():
    # Get sample output
    sample_output, info = model(X_val[:10])

    # Fusion weights
    fusion_weights = info['fusion_weights']
    print(f"\n   Learned Feature Fusion Weights:")
    for i, weight in enumerate(fusion_weights):
        print(f"      Stream {i+1}: {weight.item():.4f}")

    # Predictions
    predictions = sample_output.argmax(1)
    print(f"\n   Sample Predictions (first 10):")
    for i, (pred, true) in enumerate(zip(predictions, y_val[:10])):
        status = "✓" if pred == true else "✗"
        print(f"      {status} Sample {i+1}: Predicted={pred.item()}, True={true.item()}")

# ============================================================================
# Architecture advantages
# ============================================================================
print("\n[4] Novel Architecture Advantages")
print("   " + "-" * 66)
print("   ✓ Multi-stream processing captures diverse feature representations")
print("   ✓ Attention mechanism focuses on important features")
print("   ✓ Learnable fusion weights adapt to data")
print("   ✓ Modular design allows easy component replacement")
print("   ✓ Dropout regularization prevents overfitting")
print("   " + "-" * 66)

# ============================================================================
# Save model
# ============================================================================
print("\n[5] Saving Model")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'architecture': 'AttentionFeatureFusionNet',
    'input_dim': 20,
    'num_classes': 5
}, 'novel_architecture_checkpoint.pt')
print("   ✓ Model saved to: novel_architecture_checkpoint.pt")

print("\n" + "=" * 70)
print("Novel architecture demonstration completed!")
