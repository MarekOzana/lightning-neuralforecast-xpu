# Lightning + NeuralForecast on Intel GPU via native PyTorch XPU

Minimal reference showing that:

- Lightning can train on Intel GPU through native `torch.xpu`
- NeuralForecast can use the same XPU path through the same custom Lightning strategy
- this works without IPEX

Why this exists: Lightning still has no built-in XPU accelerator, so this repo provides the shortest working bridge.

If you only want the reusable part, copy:

- `xpu_lightning.py`

The whole point of the repo is this usage pattern:

```python
from xpu_lightning import SingleXPUStrategy

strategy=SingleXPUStrategy(device_index=0)
```

## Start here

Try the top-level minimal examples first:

```bash
uv run minimal_lightning_xpu.py
uv run minimal_neuralforecast_xpu.py
```

Then run the proof scripts:

```bash
uv run examples/test_smoke_xpu.py
uv run examples/test_lightning_xpu.py
uv run examples/test_neuralforecast_xpu.py
uv run examples/test_performance.py
```

Or run all proof scripts with pytest:

```bash
uv run pytest -s examples
```

## What to copy

Copy `xpu_lightning.py`.

It contains only:

- `NativeXPUAccelerator`
- `SingleXPUStrategy`

Use the strategy as an instance, not a string.

## Minimal Lightning usage

```python
import pytorch_lightning as L

from xpu_lightning import SingleXPUStrategy

trainer = L.Trainer(
    strategy=SingleXPUStrategy(device_index=0),
    devices=1,
)
```

See `minimal_lightning_xpu.py`.

## Minimal NeuralForecast usage

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from xpu_lightning import SingleXPUStrategy

model = NHITS(
    h=12,
    input_size=24,
    max_steps=20,
    val_check_steps=20,
    enable_progress_bar=False,
    strategy=SingleXPUStrategy(device_index=0),
    devices=1,
)

nf = NeuralForecast(models=[model], freq="1d")
nf.fit(df)
```

See `minimal_neuralforecast_xpu.py`.

## Proof scripts

The top-level files are the fastest starting point.

The `examples/` directory is the proof layer for the public claims:

- `examples/test_smoke_xpu.py`: XPU is visible and the custom classes instantiate
- `examples/test_lightning_xpu.py`: a real Lightning training loop runs on `xpu:0`
- `examples/test_neuralforecast_xpu.py`: a real NeuralForecast fit and predict run through the same XPU strategy
- `examples/test_performance.py`: the same NeuralForecast `MLP` runs on CPU and `xpu:0`, then compares runtimes and forecast closeness

## Tested here

| Package | Version |
|---|---:|
| python | 3.13 |
| torch (XPU wheel) | 2.11.0+xpu |
| pytorch-lightning | 2.6.1 |
| neuralforecast | 3.1.6 |
| polars | 1.38.1 |
| pytest | 8.4.2 |
| scipy | 1.17.1 |
| intel_extension_for_pytorch | not installed |

These are the versions used to run the examples in this repo.

## Install

Install PyTorch separately from the official XPU wheel index:

```bash
uv venv --python 3.13
# optional if you want an activated shell env:
# source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/xpu "torch==2.11.0"
uv pip install -r requirements.txt
```

## Notes

- This repo uses native `torch.xpu` directly. IPEX is not required.
- `neuralforecast==3.1.6` builds on the `pytorch_lightning` namespace, so the examples use that namespace directly.
- `neuralforecast==3.1.6` still imports `scipy` at package import time, so `scipy` stays in `requirements.txt`.

## Scope

This repo is intentionally narrow:

- native `torch.xpu`
- single-device XPU
- minimal Lightning integration
- minimal NeuralForecast integration

It is a small reference implementation, not a package.
