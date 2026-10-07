from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_emg.processing import apply_butterworth_realtime_fixed_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply only the causal fixed-point Butterworth pipeline to one CSV."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz.")
    parser.add_argument("--hp", type=float, default=20.0, help="High-pass cutoff in Hz.")
    parser.add_argument("--lp", type=float, default=None, help="Optional low-pass cutoff in Hz.")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--notch", type=float, default=50.0, help="Notch frequency in Hz.")
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument("--frac-bits", type=int, default=15)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Columns to filter; default: all CH<number> columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_butterworth_fixed.csv"
    )

    with args.input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV senza intestazione")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    channels = args.channels or [
        name for name in fieldnames
        if len(name) > 2 and name[:2].upper() == "CH" and name[2:].isdigit()
    ]
    missing = [name for name in channels if name not in fieldnames]
    if missing:
        raise ValueError(f"Colonne mancanti: {', '.join(missing)}")
    if not channels:
        raise ValueError("Nessuna colonna CH<number> trovata")

    for channel in channels:
        values = np.asarray([float(row[channel]) for row in rows], dtype=float)
        filtered = apply_butterworth_realtime_fixed_point(
            values,
            fs=args.fs,
            hp_cutoff_hz=args.hp,
            lp_cutoff_hz=args.lp,
            order=args.order,
            fractional_bits=args.frac_bits,
            notch_hz=args.notch,
            notch_q=args.notch_q,
        )
        for row, value in zip(rows, filtered):
            row[channel] = str(float(value))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Creato: {output}")
    print(
        f"Campioni: {len(rows)} | canali: {len(channels)} | Fs={args.fs:g} Hz | "
        f"HP={args.hp:g} Hz | notch={args.notch:g} Hz Q={args.notch_q:g} | "
        f"ordine={args.order} | Q*.{args.frac_bits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
