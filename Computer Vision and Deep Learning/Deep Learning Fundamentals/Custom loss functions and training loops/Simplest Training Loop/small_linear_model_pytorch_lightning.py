#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "lightning",
# ]
# ///

# uv run "/home/ptthang/CVExpectation/Computer Vision and Deep Learning/Deep Learning Fundamentals/Custom loss functions and training loops/Simplest Training Loop/small_linear_model_pytorch_lightning.py"


import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

# Define a simple linear model using PyTorch Lightning


# Define a simple linear model using PyTorch Lightning
class LinearModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, loss):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_fn = loss

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        a, b = batch
        output = self(a)
        loss = self.loss_fn(output, b)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)


# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate some random data
input_dim = 14 * 14
bs = 50
a = torch.randn((bs, input_dim)).to(device)
b = torch.randn((bs, input_dim)).to(device)

# Prepare dataset as tuple (input, target)
dataset = torch.utils.data.TensorDataset(a, b)

# Initialize the model
loss = nn.MSELoss()
model = LinearModel(input_dim, input_dim, loss).to(device)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=1,
)

# Train the model

trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=bs))

# Compute and print final loss
model.eval()
model.to(device)
with torch.no_grad():
    output = model(a)
    final_loss = loss(output, b)
    print(f"Final loss: {final_loss.item()}")
