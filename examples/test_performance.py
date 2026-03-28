from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from math import sin
from pathlib import Path
import sys
from time import perf_counter
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import pytorch_lightning as ptl
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP

from xpu_lightning import SingleXPUStrategy

logger = logging.getLogger(__name__)

H = 24
INPUT_SIZE = 96
N_SERIES = 64
N_OBS = 512
MAX_STEPS = 100
HIDDEN_SIZE = 2048
NUM_LAYERS = 3
BATCH_SIZE = 64
FORECAST_COLUMN = "MLP"
MAX_RELATIVE_MAE = 0.05


class QuietLightningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            message in record.getMessage()
            for message in (
                "Seed set to",
                "GPU available:",
                "TPU available:",
                "`Trainer.fit` stopped:",
                "try installing [litlogger]",
            )
        )


for logger_name in (
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch.utilities.rank_zero",
    "lightning_fabric.utilities.seed",
    "lightning.fabric.utilities.seed",
):
    logging.getLogger(logger_name).addFilter(QuietLightningFilter())


class FitProofCallback(ptl.Callback):
    strategy_name: str | None = None
    root_device: str | None = None
    accelerator_name: str | None = None
    model_device: str | None = None

    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        type(self).strategy_name = type(trainer.strategy).__name__
        type(self).root_device = str(trainer.strategy.root_device)
        type(self).accelerator_name = type(trainer.accelerator).__name__
        type(self).model_device = str(pl_module.device)


@dataclass
class RunResult:
    predictions: pl.DataFrame
    fit_seconds: float
    predict_seconds: float
    total_seconds: float
    strategy_name: str
    root_device: str
    accelerator_name: str
    model_device: str


def make_panel(n_series: int = N_SERIES, n_obs: int = N_OBS) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for series_idx in range(n_series):
        phase = series_idx / 5
        for step in range(n_obs):
            rows.append(
                {
                    "unique_id": f"series_{series_idx}",
                    "ds": start + timedelta(hours=step),
                    "y": (
                        10
                        + series_idx * 0.05
                        + 0.02 * step
                        + sin(step / 9 + phase)
                        + 0.3 * sin(step / 3 + phase / 2)
                    ),
                }
            )
    return pl.DataFrame(rows)


def synchronize(device: str) -> None:
    # XPU work is asynchronous, so synchronize before and after timed sections.
    if device == "xpu":
        torch.xpu.synchronize()


def run_model(device: str, df: pl.DataFrame) -> RunResult:
    FitProofCallback.strategy_name = None
    FitProofCallback.root_device = None
    FitProofCallback.accelerator_name = None
    FitProofCallback.model_device = None
    model_kwargs = dict(
        h=H,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        val_check_steps=MAX_STEPS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        windows_batch_size=1024,
        inference_windows_batch_size=1024,
        scaler_type="identity",
        random_seed=1,
        deterministic=True,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[FitProofCallback()],
    )

    if device == "cpu":
        model_kwargs.update(accelerator="cpu", devices=1)
    else:
        assert torch.xpu.is_available(), "XPU is not available on this system."
        torch.xpu.empty_cache()
        model_kwargs.update(strategy=SingleXPUStrategy(device_index=0), devices=1)

    model = MLP(**model_kwargs)
    nf = NeuralForecast(models=[model], freq="1h")

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
        synchronize(device)
        fit_start = perf_counter()
        nf.fit(df)
        synchronize(device)
        fit_end = perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The 'predict_dataloader' does not have many workers.*",
        )
        synchronize(device)
        predict_start = perf_counter()
        predictions = nf.predict().sort(["unique_id", "ds"])
        synchronize(device)
        predict_end = perf_counter()

    if device == "xpu":
        torch.xpu.empty_cache()

    return RunResult(
        predictions=predictions,
        fit_seconds=fit_end - fit_start,
        predict_seconds=predict_end - predict_start,
        total_seconds=predict_end - fit_start,
        strategy_name=FitProofCallback.strategy_name or "",
        root_device=FitProofCallback.root_device or "",
        accelerator_name=FitProofCallback.accelerator_name or "",
        model_device=FitProofCallback.model_device or "",
    )


def compare_cpu_and_xpu(log_results: bool = False) -> tuple[RunResult, RunResult, float, float]:
    df = make_panel()

    if log_results:
        logger.info("Running NeuralForecast MLP on CPU...")
    cpu_result = run_model("cpu", df)

    if log_results:
        logger.info(
            "CPU | root device: %s | accelerator: %s | fit: %.2fs | predict: %.2fs | total: %.2fs",
            cpu_result.root_device,
            cpu_result.accelerator_name,
            cpu_result.fit_seconds,
            cpu_result.predict_seconds,
            cpu_result.total_seconds,
        )
        logger.info("Running the same NeuralForecast MLP on XPU...")
    xpu_result = run_model("xpu", df)

    cpu_values = cpu_result.predictions[FORECAST_COLUMN].to_list()
    xpu_values = xpu_result.predictions[FORECAST_COLUMN].to_list()
    abs_diffs = [abs(cpu_value - xpu_value) for cpu_value, xpu_value in zip(cpu_values, xpu_values)]
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    cpu_mean_abs = sum(abs(value) for value in cpu_values) / len(cpu_values)
    relative_mae = mean_abs_diff / cpu_mean_abs

    if log_results:
        logger.info(
            "XPU | root device: %s | accelerator: %s | fit: %.2fs | predict: %.2fs | total: %.2fs",
            xpu_result.root_device,
            xpu_result.accelerator_name,
            xpu_result.fit_seconds,
            xpu_result.predict_seconds,
            xpu_result.total_seconds,
        )
        logger.info(
            "Prediction mean abs diff: %.4f | relative MAE vs CPU forecast scale: %.2f%%",
            mean_abs_diff,
            relative_mae * 100,
        )
        logger.info("XPU speedup vs CPU: %.2fx", cpu_result.total_seconds / xpu_result.total_seconds)

    return cpu_result, xpu_result, mean_abs_diff, relative_mae


def assert_comparison(cpu_result: RunResult, xpu_result: RunResult, relative_mae: float) -> None:
    assert cpu_result.root_device == "cpu"
    assert cpu_result.model_device == "cpu"
    assert xpu_result.strategy_name == "SingleXPUStrategy"
    assert xpu_result.root_device == "xpu:0"
    assert xpu_result.accelerator_name == "NativeXPUAccelerator"
    assert xpu_result.model_device == "xpu:0"
    assert cpu_result.predictions["unique_id"].equals(xpu_result.predictions["unique_id"])
    assert cpu_result.predictions["ds"].equals(xpu_result.predictions["ds"])
    assert FORECAST_COLUMN in cpu_result.predictions.columns
    assert FORECAST_COLUMN in xpu_result.predictions.columns
    assert len(cpu_result.predictions) == N_SERIES * H
    assert len(xpu_result.predictions) == N_SERIES * H
    assert cpu_result.total_seconds > 0
    assert xpu_result.total_seconds > 0
    assert relative_mae < MAX_RELATIVE_MAE


def test_performance_cpu_vs_xpu() -> None:
    cpu_result, xpu_result, _, relative_mae = compare_cpu_and_xpu()
    assert_comparison(cpu_result, xpu_result, relative_mae)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cpu_result, xpu_result, _, relative_mae = compare_cpu_and_xpu(log_results=True)
    assert_comparison(cpu_result, xpu_result, relative_mae)
