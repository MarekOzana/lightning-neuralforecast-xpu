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


class VRAMTrackingModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # A large layer to guarantee a measurable VRAM footprint (~400MB)
        self.layer = nn.Linear(10000, 10000)
        self.vram_during_fit: int = 0
        self.batch_device_type: str = "unknown"
        self.layer_device_type: str = "unknown"

    def training_step(self, batch: tuple[torch.Tensor], _: int) -> torch.Tensor:
        x = batch[0]
        self.batch_device_type = x.device.type
        self.layer_device_type = self.layer.weight.device.type
        self.vram_during_fit = torch.xpu.memory_allocated()
        return self.layer(x).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_actually_uses_xpu_vram() -> None:
    """
    Devil's Advocate Proof:
    Skeptic: "The strategy just changes the target name string to 'xpu'. PyTorch is silently executing tensors on the CPU under the hood."
    Proof: We assert that the batch was genuinely moved by Lightning to an XPU tensor, AND that actual Intel GPU VRAM bytes were allocated during training.
    """
    torch.xpu.empty_cache()
    initial_vram = torch.xpu.memory_allocated()

    model = VRAMTrackingModel()

    # Model starts on CPU before training
    assert model.layer.weight.device.type == "cpu"

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        strategy=SingleXPUStrategy(device_index=0),
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    loader = DataLoader(
        TensorDataset(torch.randn(1, 10000)),
        batch_size=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, train_dataloaders=loader)

    # Proof 1: Lightning actually mutated the model's device during training
    assert model.layer_device_type == "xpu"

    # Proof 2: Lightning passed the batch directly to XPU before training_step
    assert model.batch_device_type == "xpu"

    # Proof 3: The ultimate proof - Intel hardware actually allocated memory.
    # Strings can be spoofed, hardware VRAM counters cannot.
    assert model.vram_during_fit > initial_vram
    assert model.vram_during_fit > 0


if __name__ == "__main__":
    test_actually_uses_xpu_vram()
    print("VRAM alloc proof: PASSED (Real GPU memory was consumed!)")
