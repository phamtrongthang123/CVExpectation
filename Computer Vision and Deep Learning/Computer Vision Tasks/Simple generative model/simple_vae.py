#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple Variational Autoencoder (VAE) for generating synthetic data.
Demonstrates: encoder-decoder architecture, latent space, and reparameterization trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Create model
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Generate synthetic training data (simulating 28x28 images)
def generate_synthetic_data(batch_size=64):
    """Generate random binary images"""
    return torch.bernoulli(torch.rand(batch_size, 784))

# Training loop
print("Training VAE on synthetic data...")
model.train()

for epoch in range(10):
    # Generate batch
    data = generate_synthetic_data(batch_size=64)

    # Forward pass
    recon_batch, mu, logvar = model(data)
    loss = vae_loss(recon_batch, data, mu, logvar)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.2f}")

# Generate new samples from the latent space
print("\nGenerating new samples from latent space...")
model.eval()
with torch.no_grad():
    # Sample from standard normal distribution
    z = torch.randn(5, 20)
    samples = model.decode(z)

    print(f"Generated {len(samples)} new samples")
    print(f"Sample shape: {samples[0].shape} (can be reshaped to 28x28 image)")
    print(f"Sample statistics - Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
