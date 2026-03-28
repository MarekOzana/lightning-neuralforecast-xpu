from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class NativeXPUAccelerator(Accelerator):
    """Minimal native torch.xpu accelerator for Lightning."""

    @staticmethod
    def name() -> str:
        return "xpu"

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device}.")
        if not self.is_available():
            raise MisconfigurationException("XPU is not available on this system.")
        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return devices

    @staticmethod
    def get_parallel_devices(devices: int) -> list[torch.device]:
        return [torch.device("xpu", index) for index in range(int(devices))]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count() if NativeXPUAccelerator.is_available() else 0

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        return {}

    def teardown(self) -> None:
        if self.is_available():
            torch.xpu.empty_cache()

    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description="Native Intel XPU accelerator via torch.xpu",
        )


class SingleXPUStrategy(SingleDeviceStrategy):
    """Minimal single-device strategy for native torch.xpu."""

    strategy_name = "xpu_single"

    def __init__(self, device_index: int = 0, **kwargs: Any) -> None:
        super().__init__(
            device=torch.device("xpu", device_index),
            accelerator=NativeXPUAccelerator(),
            **kwargs,
        )

    @classmethod
    def register_strategies(cls, strategy_registry: Any) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description="Single-device Intel XPU strategy",
        )


if __name__ == "__main__":
    accelerator = NativeXPUAccelerator()
    assert accelerator.is_available(), "XPU not found. Check driver and torch XPU wheel."
    assert accelerator.auto_device_count() >= 1

    strategy = SingleXPUStrategy(device_index=0)
    assert str(strategy.root_device) == "xpu:0", strategy.root_device

    print("NativeXPUAccelerator: OK")
    print("SingleXPUStrategy: OK")
