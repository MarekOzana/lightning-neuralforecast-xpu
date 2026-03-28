from __future__ import annotations

from datetime import date, timedelta
from math import sin
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import pytorch_lightning as ptl
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from xpu_lightning import SingleXPUStrategy


def make_panel(n_series: int = 2, n_obs: int = 72) -> pl.DataFrame:
    start = date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for series_idx in range(n_series):
        for step in range(n_obs):
            rows.append(
                {
                    "unique_id": f"series_{series_idx}",
                    "ds": start + timedelta(days=step),
                    "y": 10 + series_idx + 0.1 * step + sin(step / 6),
                }
            )
    return pl.DataFrame(rows)


class FitProofCallback(ptl.Callback):
    strategy_name: str | None = None
    root_device: str | None = None
    accelerator_name: str | None = None

    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        type(self).strategy_name = type(trainer.strategy).__name__
        type(self).root_device = str(trainer.strategy.root_device)
        type(self).accelerator_name = type(trainer.accelerator).__name__
        print(
            "NeuralForecast model device:",
            pl_module.device,
            "| strategy root device:",
            trainer.strategy.root_device,
            "| accelerator:",
            type(self).accelerator_name,
        )


def test_neuralforecast_xpu() -> None:
    FitProofCallback.strategy_name = None
    FitProofCallback.root_device = None
    FitProofCallback.accelerator_name = None
    model = NHITS(
        h=12,
        input_size=24,
        max_steps=20,
        val_check_steps=20,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[FitProofCallback()],
        strategy=SingleXPUStrategy(device_index=0),
        devices=1,
    )
    nf = NeuralForecast(models=[model], freq="1d")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'train_dataloader' does not have many workers.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'val_dataloader' does not have many workers.*",
        )
        nf.fit(make_panel())

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'predict_dataloader' does not have many workers.*",
        )
        preds = nf.predict()

    assert FitProofCallback.strategy_name == "SingleXPUStrategy"
    assert FitProofCallback.root_device == "xpu:0"
    assert FitProofCallback.accelerator_name == "NativeXPUAccelerator"
    assert "NHITS" in preds.columns
    assert preds["NHITS"].null_count() == 0
    assert preds["unique_id"].n_unique() == 2
    assert len(preds) == 24


if __name__ == "__main__":
    test_neuralforecast_xpu()
    print("NeuralForecast XPU proof: PASSED")
