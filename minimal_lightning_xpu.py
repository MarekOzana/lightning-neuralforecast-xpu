"""Copyright (c) 2026 Marek Ozana, Ph.D.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import pytorch_lightning as L
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, TensorDataset

from xpu_lightning import SingleXPUStrategy

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
)
# Keep demo output focused on XPU proof instead of Lightning marketing hints.
for logger_name in (
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch.utilities.rank_zero",
):
    logging.getLogger(logger_name).addFilter(
        lambda record: "try installing [litlogger]" not in record.getMessage()
    )


class TinyModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        self.loss_fn = nn.MSELoss()

    def on_fit_start(self) -> None:
        print(
            "Lightning model device:",
            self.device,
            "| root_device:",
            self.trainer.strategy.root_device,
            "| accelerator:",
            type(self.trainer.accelerator).__name__,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x).squeeze(-1), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(128, 16), torch.randn(128)),
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    trainer = L.Trainer(
        max_epochs=2,
        strategy=SingleXPUStrategy(device_index=0),
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(TinyModel(), train_dataloaders=loader)
    print("Lightning training: PASSED")


if __name__ == "__main__":
    main()
