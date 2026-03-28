"""Copyright (c) 2026 Marek Ozana, Ph.D.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xpu_lightning import SingleXPUStrategy


class TinyModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        self.loss_fn = nn.MSELoss()
        self.fit_device: str | None = None
        self.fit_root_device: str | None = None
        self.fit_accelerator: str | None = None

    def on_fit_start(self) -> None:
        self.fit_device = str(self.device)
        self.fit_root_device = str(self.trainer.strategy.root_device)
        self.fit_accelerator = type(self.trainer.accelerator).__name__
        print(
            "Lightning model device:",
            self.device,
            "| strategy root device:",
            self.trainer.strategy.root_device,
            "| accelerator:",
            type(self.trainer.accelerator).__name__,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        return self.loss_fn(self(x).squeeze(-1), y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def test_lightning_xpu() -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(128, 16), torch.randn(128)),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    model = TinyModel()
    trainer = L.Trainer(
        max_epochs=2,
        strategy=SingleXPUStrategy(device_index=0),
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        )
        trainer.fit(model, train_dataloaders=loader)

    assert str(trainer.strategy.root_device) == "xpu:0"
    assert type(trainer.strategy).__name__ == "SingleXPUStrategy"
    assert type(trainer.accelerator).__name__ == "NativeXPUAccelerator"
    assert model.fit_device == "xpu:0"
    assert model.fit_root_device == "xpu:0"
    assert model.fit_accelerator == "NativeXPUAccelerator"
    assert trainer.global_step > 0


if __name__ == "__main__":
    test_lightning_xpu()
    print("Lightning XPU proof: PASSED")
