from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_csv(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"CH{i}" for i in range(1, data.shape[1] + 1)])
        writer.writerows(data)
