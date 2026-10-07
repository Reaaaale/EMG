from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RateMonitor:
    target_sps: float
    log_interval_s: float = 1.0
    total_started_at: float = 0.0
    total_samples: int = 0
    window_started_at: float = 0.0
    window_samples: int = 0

    def reset(self, target_sps: float | None = None) -> str:
        if target_sps is not None:
            self.target_sps = target_sps
        now = time.monotonic()
        self.total_started_at = now
        self.total_samples = 0
        self.window_started_at = now
        self.window_samples = 0
        return f"[ACQ] START target={self.target_sps:.0f} SPS, log_window={self.log_interval_s:.1f}s"

    def record_sample(self) -> str | None:
        self.total_samples += 1
        self.window_samples += 1
        now = time.monotonic()
        elapsed = now - self.window_started_at
        if elapsed < self.log_interval_s:
            return None

        actual_sps = self.window_samples / elapsed
        expected_samples = self.target_sps * elapsed
        missing_estimate = expected_samples - self.window_samples
        error_percent = self._error_percent(actual_sps)

        total_elapsed = now - self.total_started_at
        total_expected = self.target_sps * total_elapsed
        total_missing_estimate = total_expected - self.total_samples

        self.window_started_at = now
        self.window_samples = 0
        return (
            "[ACQ] "
            f"window={elapsed:.2f}s "
            f"target={self.target_sps:.0f}SPS "
            f"actual={actual_sps:.1f}SPS "
            f"err={error_percent:+.1f}% "
            f"window_missing~={missing_estimate:.0f} "
            f"total_samples={self.total_samples} "
            f"total_missing~={total_missing_estimate:.0f}"
        )

    def summary(self) -> str | None:
        if not self.total_started_at:
            return None
        elapsed = time.monotonic() - self.total_started_at
        if elapsed <= 0:
            return None
        actual_sps = self.total_samples / elapsed
        expected_samples = self.target_sps * elapsed
        missing_estimate = expected_samples - self.total_samples
        error_percent = self._error_percent(actual_sps)
        return (
            "[ACQ] STOP "
            f"duration={elapsed:.2f}s "
            f"target={self.target_sps:.0f}SPS "
            f"actual_avg={actual_sps:.1f}SPS "
            f"err={error_percent:+.1f}% "
            f"samples={self.total_samples} "
            f"missing~={missing_estimate:.0f}"
        )

    def _error_percent(self, actual_sps: float) -> float:
        if not self.target_sps:
            return 0.0
        return 100 * (actual_sps - self.target_sps) / self.target_sps

