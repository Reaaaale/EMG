from __future__ import annotations

from collections import deque

import numpy as np


class RingBuffer:
    def __init__(self, window_size: int, n_channels: int) -> None:
        self.window_size = int(window_size)
        self.n_channels = int(n_channels)
        self._samples: deque[np.ndarray] = deque(maxlen=self.window_size)

    def clear(self) -> None:
        self._samples.clear()

    def add(self, sample: np.ndarray) -> None:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (self.n_channels,):
            raise ValueError(f"Expected sample shape {(self.n_channels,)}, got {sample.shape}")
        self._samples.append(sample.copy())

    def ready(self) -> bool:
        return len(self._samples) == self.window_size

    def get_window(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("Buffer is not full yet")
        return np.stack(list(self._samples), axis=0).astype(np.float32)
