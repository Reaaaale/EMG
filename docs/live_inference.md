# Live Inference

This document describes how the desktop app performs live EMG inference while samples are arriving from the ADS board.

## Runtime Flow

```text
ADS serial sample
  -> EEGEMGApp.append_sample(sample)
  -> LiveInferencePipeline.add_sample(sample[:8])
  -> RingBuffer keeps the latest window_size samples
  -> every window_shift samples, get_window()
  -> encode_window()
  -> model(x) with batch_size = 1
  -> scores = output.sum(dim=2)
  -> predicted_class = scores.argmax(dim=1)
  -> UI update + optional CSV export
```

## Windowing

The live model does not classify one raw ADS sample directly. It classifies a short window of recent samples.

Current defaults are:

```text
window_size = 100 samples
window_shift = 25 samples
n_channels = 8
delta = 500
```

At 1 kHz this means:

```text
window_size = 100 ms of context
window_shift = one prediction every 25 ms
```

The windows overlap. With `window_size=100` and `window_shift=25`, consecutive predictions share 75 samples.

## Batch Size

Live inference uses `batch_size = 1`.

Before inference, `encode_window()` returns spikes with shape:

```text
features x time
```

The model expects a batch dimension:

```text
batch x features x time
```

So the live code uses:

```python
x = torch.tensor(spikes[None, :, :], dtype=torch.float32, device=device)
```

If `spikes.shape == (48, 92)`, then:

```text
x.shape == (1, 48, 92)
```

The leading `1` is the live batch size: one encoded window at a time.

## Encoding

`encode_window()` converts a numeric EMG window into spike features.

With derivatives disabled and 8 channels, the spike features are:

```text
8 channels * 2 polarities = 16 features
```

With derivatives up to second order, the features are:

```text
8 channels * 3 orders * 2 polarities = 48 features
```

The time dimension can become shorter than `window_size` when derivatives are enabled because samples at the borders are trimmed.
For example, a 100-sample window can become 92 time steps.

## Classification

The network output has shape:

```text
batch x num_classes x time
```

Classification uses rate decoding:

```python
scores = output.sum(dim=2)
pred = scores.argmax(dim=1)
```

This is equivalent in intent to `slayer.classifier.Rate.predict(output)` used during validation.

## Important Files

```text
src/eeg_emg/app.py                  UI, serial acquisition, replay, live status
src/eeg_emg/inference/pipeline.py   Live inference orchestration
src/eeg_emg/inference/buffer.py     Ring buffer for live windows
src/eeg_emg/inference/encoding.py   Delta encoding and derivative expansion
src/eeg_emg/inference/model.py      Network compatibility and model loading
src/eeg_emg/replay.py               CSV replay through the same live path
```

## Model Loading Note

Some older checkpoints were saved from a notebook as a full PyTorch model. In that case PyTorch looks for a class named `__main__.Network` while loading. The app registers a compatible `Network` class before `torch.load()` so those checkpoints can still be opened.

For new work, prefer saving:

```python
torch.save(model.state_dict(), path)
```

with a matching `config.json` next to the checkpoint.

## Training in this repository

See [`notebook/training.ipynb`](../notebook/training.ipynb) and
[notebook setup](../notebook/README.md). Install `.[inference]` on Python 3.10
for model loading. Select `model_state_dict.pt` from a training run and keep
`config.json` in the same directory. The configuration overrides the fallback
window and encoding defaults described above (the included notebook currently
uses a 300-sample window).

Share the entire checkpoint/config pair, not a renamed checkpoint separated
from its configuration. Store local runs in ignored `results/` or downloaded
models in ignored `models/`. Filtered datasets also require matching live
preprocessing; inspect filter settings and avoid filtering replay data twice.
