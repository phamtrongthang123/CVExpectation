#!/usr/bin/env python3
import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class LargeModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, model_size="medium"):
        super().__init__()
        self.save_hyperparameters()

        # Create different model sizes to test DeepSpeed benefits
        if model_size == "small":
            hidden_sizes = [512, 256, 128]
        elif model_size == "medium":
            hidden_sizes = [2048, 1024, 512]
        elif model_size == "large":
            hidden_sizes = [4096, 2048, 1024]
        else:
            hidden_sizes = [8192, 4096, 2048, 1024]  # xlarge

        # Build a larger model that benefits from DeepSpeed
        layers = []
        in_features = 28 * 28

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.LayerNorm(hidden_size),  # LayerNorm works better with DeepSpeed
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            in_features = hidden_size

        self.model = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_sizes[-1], 10)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        linear_features = self.model(x_flat)

        return self.classifier(linear_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log learning rate (important for DeepSpeed)
        if self.trainer.is_global_zero:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", current_lr, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # DeepSpeed will override this, but we still need to define it
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # Download only on rank 0
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            datasets.MNIST("./data", train=True, download=True)
            datasets.MNIST("./data", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST("./data", train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument("--deepspeed_stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        help="Offload optimizer to CPU (Stage 2+)",
    )
    parser.add_argument(
        "--offload_params",
        action="store_true",
        help="Offload parameters to CPU (Stage 3)",
    )
    parser.add_argument("--ckpt_path", default="", type=str, help="Path to checkpoint")
    args = parser.parse_args()

    # Initialize model and data
    model = LargeModel(learning_rate=args.learning_rate, model_size=args.model_size)
    data_module = DataModule(
        batch_size=args.batch_size,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    )

    # DeepSpeed configuration
    deepspeed_config = {
        "zero_optimization": {
            "stage": args.deepspeed_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 200000000,
            "contiguous_gradients": True,
        },
        # this will override what i have in lightning optimizer and scheduler.
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
        },
        "train_micro_batch_size_per_gpu": args.batch_size, # batch size per gpu, don't set total batch size here, let deep speed handle it.
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": True}, # i love bf16
    }

    # Add CPU offloading options
    if args.deepspeed_stage >= 2 and args.offload_optimizer:
        deepspeed_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if args.deepspeed_stage >= 3 and args.offload_params:
        deepspeed_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    # Initialize DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=args.deepspeed_stage,
        offload_optimizer=args.offload_optimizer
        if args.deepspeed_stage >= 2
        else False,
        offload_parameters=args.offload_params if args.deepspeed_stage >= 3 else False,
        config=deepspeed_config,
        logging_level=20,  # INFO level
    )

    # Get number of nodes from SLURM environment
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,  # DeepSpeed manages device assignment
        num_nodes=num_nodes,  # Explicitly set from SLURM
        strategy=strategy,
        precision="bf16-mixed", 
        # Important: DeepSpeed handles its own checkpointing
        enable_checkpointing=True,
        default_root_dir="./deepspeed_logs",
        # Logging
        log_every_n_steps=20,
        enable_progress_bar=True,
        enable_model_summary=False, # not sure why but there is a bug at this moment of this codebase.
        # Callbacks
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=2,
                filename="deepspeed-{epoch:02d}-{val_acc:.2f}",
                every_n_epochs=5,  # Less frequent saves with DeepSpeed
            ),
            pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=5),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    # Print configuration info
    if trainer.is_global_zero:
        print(f"Number of Nodes: {num_nodes}")
        print(f"DeepSpeed Stage: {args.deepspeed_stage}")
        print(f"Model Size: {args.model_size}")
        print(f"World Size: {trainer.world_size}")
        print(f"Global Batch Size: {args.batch_size * trainer.world_size}")
        print(f"Optimizer Offload: {args.offload_optimizer}")
        print(f"Parameter Offload: {args.offload_params}")

        # Print model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")

    # Start training. It it easy to just load ckpt path here, even if it is a folder from deepspeed.
    if args.ckpt_path != "":
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
