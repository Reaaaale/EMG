from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .buffer import RingBuffer
from .encoding import encode_window
from .model import (
    Network,
    NetworkBuilder,
    config_input_features,
    config_max_derivative_order,
    config_num_classes,
    read_training_config,
)
from ..config import MIN_LIVE_INFERENCE_WINDOW_SHIFT
from ..processing import StreamingButterworthFilter


@dataclass
class InferenceResult:
    """Result produced from one live inference window."""

    sample_index: int
    predicted_class: int | None
    class_name: str
    scores: np.ndarray | None
    encoding_ms: float
    inference_ms: float
    total_ms: float
    spike_rate: float


class LiveInferencePipeline:
    """Live inference pipeline used by the Qt app.

    The network does not classify individual ADS samples. Incoming samples are
    accumulated in a ring buffer; every `window_shift` samples, the latest
    `window_size` samples are encoded into spikes and passed to the model.
    """

    def __init__(
        self,
        window_size: int = 100,
        window_shift: int = 25,
        n_channels: int = 8,
        delta: float = 500.0,
        max_derivative_order: int = 0,
        encoding_mode: str = "standard",
        lambda_d: float = 0.001,
        delta_by_order: list[float] | None = None,
        filter_enabled: bool = True,
        filter_fs: float = 1000.0,
        filter_hp_hz: float | None = 5.0,
        filter_lp_hz: float | None = None,
        filter_order: int = 4,
        filter_notch_hz: float | None = 50.0,
        filter_notch_q: float = 30.0,
        class_names: dict[int, str] | None = None,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        # Number of raw samples used for one prediction. At 1 kHz, 100 samples
        # represent 100 ms of context.
        self.window_size = int(window_size)

        # Prediction cadence. At 1 kHz, 25 samples means one prediction every
        # 25 ms, with overlapping windows.
        self.window_shift = int(window_shift)

        # Only the first 8 EMG channels are currently used. CH9-CH16 are empty
        # in the present acquisition setup.
        self.n_channels = int(n_channels)

        # Delta encoding threshold: lower values generate denser spike trains.
        self.delta = float(delta)

        # Derivative feature expansion order. 0 means raw channels only.
        self.max_derivative_order = int(max_derivative_order)
        self.encoding_mode = str(encoding_mode).strip().lower()
        self.lambda_d = float(lambda_d)
        self.delta_by_order = (
            [float(value) for value in delta_by_order]
            if delta_by_order is not None
            else None
        )
        self.filter_enabled = bool(filter_enabled)
        self.filter_fs = float(filter_fs)
        self.filter_hp_hz = filter_hp_hz
        self.filter_lp_hz = filter_lp_hz
        self.filter_order = int(filter_order)
        self.filter_notch_hz = filter_notch_hz
        self.filter_notch_q = float(filter_notch_q)
        self.class_names = class_names or {0: "rest", 1: "sasso", 2: "forbici", 3: "carta"}
        self.device = device

        
        self.buffer = RingBuffer(self.window_size, self.n_channels)
        self._build_streaming_filter()
        self.sample_index = 0
        self.model = None
        self.torch = None
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str | Path) -> None:
        import sys

        import torch

        path = Path(model_path)
        config = read_training_config(path)

        
        self.window_size = int(config.get("window_size", self.window_size))
        self.window_shift = max(
            int(config.get("window_shift", self.window_shift)),
            MIN_LIVE_INFERENCE_WINDOW_SHIFT,
        )
        self.delta = float(config.get("delta", self.delta))
        self.max_derivative_order = config_max_derivative_order(config, n_channels=self.n_channels)
        self.encoding_mode = str(config.get("encoding_mode", "standard")).strip().lower()
        self.lambda_d = float(config.get("lambda_d", self.lambda_d))
        configured_deltas = config.get("delta_by_order")
        self.delta_by_order = (
            [float(value) for value in configured_deltas]
            if configured_deltas is not None
            else [self.delta] * (self.max_derivative_order + 1)
        )
        if "filter_enabled" in config:
            self.filter_enabled = bool(config["filter_enabled"])
        elif "data_version" in config:
            self.filter_enabled = str(config["data_version"]).lower() == "filtered"
        self.filter_fs = float(config.get("filter_fs", config.get("sample_rate_hz", self.filter_fs)))
        self.filter_hp_hz = config.get("filter_hp_hz", self.filter_hp_hz)
        self.filter_lp_hz = config.get("filter_lp_hz", self.filter_lp_hz)
        self.filter_order = int(config.get("filter_order", self.filter_order))
        self.filter_notch_hz = config.get("filter_notch_hz", self.filter_notch_hz)
        self.filter_notch_q = float(config.get("filter_notch_q", self.filter_notch_q))
        config_class_names = config.get("class_names")
        if isinstance(config_class_names, list):
            self.class_names = {
                index: str(name) for index, name in enumerate(config_class_names)
            }
        elif isinstance(config_class_names, dict):
            self.class_names = {
                int(index): str(name) for index, name in config_class_names.items()
            }

        # Recreate the buffer if the loaded model changes the window settings.
        self.buffer = RingBuffer(self.window_size, self.n_channels)
        self._build_streaming_filter()
        self.sample_index = 0

        self.torch = torch

        # Old notebook checkpoints saved the whole model as __main__.Network.
        # Register a compatible class so those files can still be opened from
        # the installed app. This does not affect inference outputs.
        setattr(sys.modules["__main__"], "Network", Network)
        loaded = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(loaded, dict):
            input_features = config_input_features(config, fallback=self.n_channels * 2)
            num_classes = config_num_classes(config, fallback=len(self.class_names))
            model = NetworkBuilder(input_features=input_features, num_classes=num_classes).build()
            model.load_state_dict(loaded)
            self.model = model
        else:
            self.model = loaded

        self.model.to(self.device)
        self.model.eval()

    def clear(self) -> None:
        self.buffer.clear()
        if self.streaming_filter is not None:
            self.streaming_filter.reset()
        self.sample_index = 0

    def set_filter_enabled(self, enabled: bool) -> None:
        """Enable/disable preprocessing without mixing unlike samples."""
        self.filter_enabled = bool(enabled)
        self._build_streaming_filter()
        self.buffer.clear()

    def _build_streaming_filter(self) -> None:
        if not self.filter_enabled:
            self.streaming_filter = None
            return
        self.streaming_filter = StreamingButterworthFilter(
            n_channels=self.n_channels,
            fs=self.filter_fs,
            hp_cutoff_hz=self.filter_hp_hz,
            lp_cutoff_hz=self.filter_lp_hz,
            order=self.filter_order,
            notch_hz=self.filter_notch_hz,
            notch_q=self.filter_notch_q,
        )

    def add_sample(self, sample: np.ndarray) -> InferenceResult | None:
        """Add one live sample and return a prediction when a window is ready.

        Most calls return None: first while the buffer is filling, then between
        two inference steps. An InferenceResult is returned every `window_shift`
        samples once the ring buffer contains a full window.
        """

        self.sample_index += 1

        # One call corresponds to one ADS sample. Keep the first `n_channels`
        # values and append them to the temporal buffer.
        sample = np.asarray(sample, dtype=np.float32)[: self.n_channels]
        if self.streaming_filter is not None:
            sample = self.streaming_filter.process_sample(sample)
        self.buffer.add(sample)

        # No inference is possible until we have `window_size` samples.
        if not self.buffer.ready():
            return None

        # Produce predictions only at the configured stride, not at every sample.
        if self.sample_index % self.window_shift != 0:
            return None

        total_start = time.perf_counter()

        # Shape: (window_size, n_channels), for example (100, 8).
        window = self.buffer.get_window()

        enc_start = time.perf_counter()
        # Convert the numeric EMG window into spikes. Shape is (features, time).
        spikes = encode_window(
            window,
            delta=self.delta,
            max_derivative_order=self.max_derivative_order,
            mode=self.encoding_mode,
            lambda_d=self.lambda_d,
            delta_by_order=self.delta_by_order,
        )
        encoding_ms = (time.perf_counter() - enc_start) * 1000
        spike_rate = float(spikes.mean())

        predicted_class = None
        class_name = "encoding_only"
        scores_np = None
        inference_ms = 0.0

        if self.model is not None and self.torch is not None:
            infer_start = time.perf_counter()
            with self.torch.no_grad():
                # Live batch size is 1. If spikes is (features, time), adding
                # the leading dimension makes x shape (1, features, time).
                x = self.torch.tensor(spikes[None, :, :], dtype=self.torch.float32, device=self.device)

                # Output shape is (batch, num_classes, time).
                output = self.model(x)

                # Rate decoding: sum over time to get one score per class,
                # then choose the class with the highest score.
                scores = output.sum(dim=2)
                pred = scores.argmax(dim=1)
            inference_ms = (time.perf_counter() - infer_start) * 1000

            # Convert tensors back to Python/Numpy values for the UI and CSV log.
            predicted_class = int(pred.detach().cpu().item())
            class_name = self.class_names.get(predicted_class, str(predicted_class))
            scores_np = scores.detach().cpu().numpy()[0]

        total_ms = (time.perf_counter() - total_start) * 1000
        return InferenceResult(
            sample_index=self.sample_index,
            predicted_class=predicted_class,
            class_name=class_name,
            scores=scores_np,
            encoding_ms=encoding_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
            spike_rate=spike_rate,
        )
