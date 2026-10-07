from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from .config import CHANNEL_COUNT


@dataclass
class ReplayStats:
    elapsed_s: float
    emitted_samples: int
    target_sps: float

    @property
    def actual_sps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.emitted_samples / self.elapsed_s


class CsvReplay(QtCore.QObject):
    sample_ready = QtCore.Signal(object)
    status = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._emit_due_samples)
        self.samples = np.empty((0, CHANNEL_COUNT), dtype=np.float32)
        self.labels: np.ndarray | None = None
        self.path: Path | None = None
        self.fs = 1000.0
        self.index = 0
        self.started_at = 0.0
        self.last_status_at = 0.0

    def load_csv(self, path: str | Path) -> None:
        path = Path(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV senza header")
            rows = list(reader)

        if not rows:
            raise ValueError("CSV vuoto")

        channels = [name for name in reader.fieldnames if name.upper().startswith("CH")]
        if not channels:
            raise ValueError("Nessuna colonna CH trovata")

        channels = sorted(channels, key=_channel_sort_key)
        data = np.zeros((len(rows), CHANNEL_COUNT), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for col_idx, name in enumerate(channels[:CHANNEL_COUNT]):
                value = row.get(name, "")
                try:
                    data[row_idx, col_idx] = float(value)
                except ValueError:
                    data[row_idx, col_idx] = np.nan

        labels = None
        if "label" in reader.fieldnames:
            labels = np.array([row.get("label", "") for row in rows], dtype=object)

        self.samples = data
        self.labels = labels
        self.path = path
        self.index = 0
        self.status.emit(f"Replay CSV caricato: {path.name} ({len(data)} campioni, {len(channels)} canali)")

    def start(self, fs: float) -> None:
        if self.samples.size == 0:
            raise ValueError("Carica prima un CSV")
        self.fs = float(fs)
        if self.fs <= 0:
            raise ValueError("Fs replay non valida")
        self.index = 0
        self.started_at = time.perf_counter()
        self.last_status_at = self.started_at
        self.timer.start(1)
        self.status.emit(f"Replay START target={self.fs:.0f} SPS")

    def stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        stats = self.stats()
        self.status.emit(
            f"Replay STOP duration={stats.elapsed_s:.2f}s "
            f"target={stats.target_sps:.0f}SPS actual={stats.actual_sps:.1f}SPS "
            f"samples={stats.emitted_samples}"
        )
        self.finished.emit()

    def is_running(self) -> bool:
        return self.timer.isActive()

    def stats(self) -> ReplayStats:
        elapsed = 0.0 if not self.started_at else time.perf_counter() - self.started_at
        return ReplayStats(elapsed_s=elapsed, emitted_samples=self.index, target_sps=self.fs)

    def _emit_due_samples(self) -> None:
        if self.samples.size == 0 or not self.started_at:
            return

        elapsed = time.perf_counter() - self.started_at
        target_index = min(int(elapsed * self.fs), len(self.samples))

        while self.index < target_index:
            self.sample_ready.emit(self.samples[self.index].copy())
            self.index += 1

        now = time.perf_counter()
        if now - self.last_status_at >= 1.0:
            self.last_status_at = now
            stats = self.stats()
            self.status.emit(
                f"Replay window target={stats.target_sps:.0f}SPS "
                f"actual_avg={stats.actual_sps:.1f}SPS "
                f"sample={self.index}/{len(self.samples)}"
            )

        if self.index >= len(self.samples):
            self.stop()


def _channel_sort_key(name: str) -> tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits), name
    return 10_000, name
