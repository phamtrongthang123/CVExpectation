#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
# ]
# ///

# uv run "/home/ptthang/CVExpectation/Computer Vision and Deep Learning/Deep Learning Fundamentals/Custom loss functions and training loops/Simplest Training Loop/simplest_training_loop.py"
import torch
from torch import nn
from torch.optim import Adam

a = nn.Parameter(torch.randn((7, 14, 14), requires_grad=True))
b = torch.randn((7, 14, 14))
optimizer = Adam([a], lr=0.1)
loss = nn.MSELoss()

for i in range(10000):
    optimizer.zero_grad()
    loss_value = loss(a, b)
    loss_value.backward()
    optimizer.step()
    print(loss_value.item())
