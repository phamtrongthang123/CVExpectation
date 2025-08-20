#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "lightning",
# ]
# ///

# uv run small_linear_model_pytorch_lightning.py


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==========================================================
# 1. MODEL (YOU MUST CHANGE)
# ==========================================================
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 10)  # 👈 replace with your architecture

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):  # 👈 test loop
        x, y = batch
        logits = self(x)
        test_loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ==========================================================
# 2. DATASETS (YOU MUST CHANGE)
# ==========================================================
class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.x = torch.randn(size, 32)
        self.y = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_loader = DataLoader(MyDataset(1000), batch_size=32, shuffle=True)
val_loader = DataLoader(MyDataset(200), batch_size=32)
test_loader = DataLoader(MyDataset(200), batch_size=32)

# ==========================================================
# 3. TRAINER
# ==========================================================
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,  # 👈 set to 8 for multi-GPU
    max_epochs=5,
)

# ==========================================================
# 4. FIT + TEST
# ==========================================================
model = MyModel()
trainer.fit(model, train_loader, val_loader)

# After training, run test loop
trainer.test(model, test_loader)  # 👈 Lightning auto-runs test_step
