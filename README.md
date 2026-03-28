# Lightning + NeuralForecast on Intel GPU — Native PyTorch XPU

- Use `SingleXPUStrategy(device_index=0)` to run Lightning and NeuralForecast on native `torch.xpu`.

That's it. Tested here on Intel Arc Pro B50 / Ubuntu 25.10 and Windows 11 with `torch==2.11.0+xpu`, `pytorch-lightning==2.6.1`, and `neuralforecast==3.1.6`.

## Choose your path

**→ I just want the fix** — skip to [Copy one file](#copy-one-file)
**→ I want to run a working example first** — skip to [Run the examples](#run-the-examples)

## Copy one file

Copy [`xpu_lightning.py`](./xpu_lightning.py).

- `NativeXPUAccelerator` subclasses `Accelerator` and maps Lightning's device hooks to native `torch.xpu`.
- `SingleXPUStrategy` subclasses `SingleDeviceStrategy` and builds `torch.device("xpu", device_index)` around `NativeXPUAccelerator()`.

### Minimal Lightning usage
```python
import pytorch_lightning as L

from xpu_lightning import SingleXPUStrategy

trainer = L.Trainer(
    strategy=SingleXPUStrategy(device_index=0),
    devices=1,
)
```

### Minimal NeuralForecast usage
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from xpu_lightning import SingleXPUStrategy

model = NHITS(
    h=12,
    input_size=24,
    max_steps=20,
    strategy=SingleXPUStrategy(device_index=0),
    devices=1,
)
nf = NeuralForecast(models=[model], freq="1d")
nf.fit(df)
```

See [minimal_lightning_xpu.py](./minimal_lightning_xpu.py) and [minimal_neuralforecast_xpu.py](./minimal_neuralforecast_xpu.py).

## Run the examples

### Install

```bash
# ================================================
# DO NOT INSTALL intel_extension_for_pytorch HERE.
# This repo uses native torch.xpu only.
# ================================================

pip install --index-url https://download.pytorch.org/whl/xpu "torch==2.11.0"
pip install "pytorch-lightning==2.6.1" "neuralforecast==3.1.6" "polars==1.38.1" "pytest==8.4.2" "scipy==1.17.1"
```

Or with uv for the requirements file: `uv pip install -r requirements.txt`

### Run

```bash
python minimal_lightning_xpu.py
# or: uv run minimal_lightning_xpu.py
```

```bash
python minimal_neuralforecast_xpu.py
# or: uv run minimal_neuralforecast_xpu.py
```

```bash
python -m pytest -s examples/
# or: uv run pytest -s examples/
```

## Why this works — the one insight

Lightning checks `accelerator` by name, so this is the error you hit first:

```text
ValueError: invalid accelerator name: xpu
```

Passing `SingleXPUStrategy(device_index=0)` sidesteps that string check. Lightning gets a ready-made strategy instance and follows its native `torch.device("xpu", 0)` path instead.

That whole bridge lives in `xpu_lightning.py`. The two classes in that file are 87 lines total.

## Version requirements

| Package | Tested | Tested |
| --- | --- | --- |
| OS | Ubuntu 25.10 | Windows 11 |
| Python | 3.13.0 | 3.12.13 |
| torch (xpu wheel) | 2.11.0+xpu | 2.11.0+xpu |
| pytorch_lightning | 2.6.1 | 2.6.1 |
| neuralforecast | 3.1.6 | 3.1.6 |
| polars | 1.38.1 | 1.38.1 |
| intel_extension_for_pytorch | not installed | not installed |

These are the versions tested in this repo. Other versions may work, but they are not verified here.
NeuralForecast currently publishes Python 3.10-3.13 classifiers, and Python 3.14 is not yet supported there.

## Repo structure

```text
xpu_lightning.py                ← copy this; accelerator + strategy
minimal_lightning_xpu.py        Lightning demo on xpu:0
minimal_neuralforecast_xpu.py   NeuralForecast demo on xpu:0
examples/
  test_performance.py           CPU vs XPU MLP comparison
  test_lightning_xpu.py         Lightning fit on xpu:0
  test_smoke_xpu.py             XPU and class smoke test
  test_neuralforecast_xpu.py    NHITS fit and predict on xpu:0
requirements.txt                pinned versions + XPU wheel note
```

## Context

Lightning has no built-in XPU accelerator yet; see [issue #20938](https://github.com/Lightning-AI/pytorch-lightning/issues/20938).
PyTorch has supported Intel GPUs natively since 2.5 (October 2024).
Intel EOL'd IPEX in March 2026. This repo is the shortest working bridge.

## License

MIT
