"""Copyright (c) 2026 Marek Ozana, Ph.D.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from xpu_lightning import NativeXPUAccelerator, SingleXPUStrategy


def test_smoke_xpu() -> None:
    assert hasattr(torch, "xpu"), "This torch build does not expose torch.xpu."
    assert torch.xpu.is_available(), "XPU is not available on this system."

    accelerator = NativeXPUAccelerator()
    strategy = SingleXPUStrategy(device_index=0)

    assert accelerator.is_available()
    assert accelerator.name() == "xpu"
    assert type(strategy.accelerator).__name__ == "NativeXPUAccelerator"
    assert str(strategy.root_device) == "xpu:0"


if __name__ == "__main__":
    test_smoke_xpu()
    print("Accelerator: NativeXPUAccelerator | root_device: xpu:0")
    print("XPU smoke test: PASSED")
