#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "accelerate",
# ]
# ///

# uv run small_linear_model_accelerate.py
# to enable distributed training, I must go with accelerate launch. Because accelerate is its own cli tool, actually you would need to install a small env to run this script.
# call the commands below:
# uv init
# uv add torch accelerate
# rm hello.py README.md
# source .venv/bin/activate
# Set `accelerate config` first . This will store a config for running this computer as a node.
# then call (for example run on 2 GPUs): accelerate launch --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
# To be honest, I would rather use pytorch lightning when possible. Then move to accelerate later.

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset


# ==========================================================
# 1. MODEL (same as before)
# ==========================================================
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 10)  # 👈 same architecture

    def forward(self, x):
        return self.layer(x)


# ==========================================================
# 2. DATASETS (same as before)
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
# 3. ACCELERATOR + PREP
# ==========================================================
accelerator = Accelerator()
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader, test_loader
)

# ==========================================================
# 4. TRAIN LOOP
# ==========================================================
for epoch in range(5):
    model.train()
    for batch in train_loader:
        x, y = batch
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    # ----- Validation -----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            val_loss += F.cross_entropy(logits, y, reduction="sum").item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    if accelerator.is_main_process:
        print(
            f"Epoch {epoch}: val_loss={val_loss / val_total:.4f}, val_acc={val_correct / val_total:.4f}"
        )

# ==========================================================
# 5. TEST LOOP
# ==========================================================
model.eval()
test_loss, test_correct, test_total = 0, 0, 0
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        test_loss += F.cross_entropy(logits, y, reduction="sum").item()
        preds = logits.argmax(dim=1)
        test_correct += (preds == y).sum().item()
        test_total += y.size(0)

if accelerator.is_main_process:
    print(
        f"TEST: loss={test_loss / test_total:.4f}, acc={test_correct / test_total:.4f}"
    )
