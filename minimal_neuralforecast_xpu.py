from __future__ import annotations

from datetime import date, timedelta
import logging
from math import sin
import warnings

import polars as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

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


def make_panel(n_series: int = 2, n_obs: int = 72) -> pl.DataFrame:
    start = date(2024, 1, 1)
    return pl.DataFrame(
        [
            {
                "unique_id": f"series_{series_idx}",
                "ds": start + timedelta(days=step),
                "y": 10 + series_idx + 0.1 * step + sin(step / 6),
            }
            for series_idx in range(n_series)
            for step in range(n_obs)
        ]
    )


def main() -> None:
    df = make_panel()

    model = NHITS(
        h=12,
        input_size=24,
        max_steps=20,
        val_check_steps=20,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        strategy=SingleXPUStrategy(device_index=0),
        devices=1,
    )
    nf = NeuralForecast(models=[model], freq="1d")

    print("Fitting NHITS on XPU...")
    nf.fit(df)

    preds = nf.predict()

    print(preds.head())
    print("NeuralForecast training and prediction: PASSED")


if __name__ == "__main__":
    main()
