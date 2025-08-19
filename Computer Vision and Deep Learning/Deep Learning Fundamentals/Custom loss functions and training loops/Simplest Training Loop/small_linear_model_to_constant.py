#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
# ]
# ///

# uv run "/home/ptthang/CVExpectation/Computer Vision and Deep Learning/Deep Learning Fundamentals/Custom loss functions and training loops/Simplest Training Loop/small_linear_model_to_constant.py"

import torch
from torch import nn
from torch.optim import Adam


# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
input_dim = 14 * 14
output_dim = 14 * 14
model = LinearModel(input_dim, output_dim).to(device)
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)

# Generate some random data
a = torch.randn((7, input_dim)).to(device)
b = torch.randn((7, output_dim)).to(device)

# Training loop
for i in range(10000):
    optimizer.zero_grad()
    output = model(a)
    loss = loss_fn(output, b)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:  # Print loss every 100 iterations
        print(f"Iteration {i}, Loss: {loss.item()}")