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

from eeg_emg.processing import apply_fir_offline_zero_phase, signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter all CSV files in a folder with the offline FIR notch + high-pass pipeline."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where filtered CSV files are written.")
    parser.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz.")
    parser.add_argument("--hp", type=float, default=5.0, help="High-pass FIR cutoff in Hz.")
    parser.add_argument("--taps", type=int, default=51, help="FIR tap count for both notch and high-pass filters.")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch center frequency in Hz.")
    parser.add_argument("--notch-bw", type=float, default=2.0, help="Notch transition bandwidth in Hz.")
    parser.add_argument("--pattern", default="*.csv", help="Input file glob pattern.")
    parser.add_argument("--recursive", action="store_true", help="Search input files recursively.")
    parser.add_argument("--suffix", default="_fir_offline", help="Suffix appended before .csv.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Columns to filter. Default: every column named CH<number>.",
    )
    return parser.parse_args()


def default_channel_names(fieldnames: list[str]) -> list[str]:
    channels: list[str] = []
    for name in fieldnames:
        if len(name) > 2 and name[:2].upper() == "CH" and name[2:].isdigit():
            channels.append(name)
    return channels


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("header CSV mancante")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_file(
    path: Path,
    output_path: Path,
    fs: float,
    hp: float,
    taps: int,
    notch: float,
    notch_bw: float,
    requested_channels: list[str] | None,
) -> tuple[int, list[str]]:
    fieldnames, rows = read_csv(path)
    if not rows:
        write_csv(output_path, fieldnames, rows)
        return 0, []

    channels = requested_channels if requested_channels is not None else default_channel_names(fieldnames)
    missing = [name for name in channels if name not in fieldnames]
    if missing:
        raise ValueError(f"colonne non trovate: {', '.join(missing)}")

    filtered_count = 0
    skipped: list[str] = []
    for channel in channels:
        try:
            x = np.array([float(row[channel]) for row in rows], dtype=float)
            y = apply_fir_offline_zero_phase(
                x,
                fs=fs,
                hp_cutoff_hz=hp,
                numtaps=taps,
                notch_hz=notch,
                notch_bw_hz=notch_bw,
            )
        except Exception as exc:
            skipped.append(f"{channel} ({exc})")
            continue

        for row, value in zip(rows, y):
            row[channel] = f"{value:.12g}"
        filtered_count += 1

    write_csv(output_path, fieldnames, rows)
    return filtered_count, skipped


def iter_input_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def output_path_for(input_path: Path, input_dir: Path, output_dir: Path, suffix: str) -> Path:
    relative = input_path.relative_to(input_dir)
    return output_dir / relative.with_name(f"{relative.stem}{suffix}{relative.suffix}")


def main() -> int:
    args = parse_args()
    if signal is None:
        print("Errore: scipy non installato, filtro non disponibile.", file=sys.stderr)
        return 2
    if not args.input_dir.exists():
        print(f"Errore: cartella input non trovata: {args.input_dir}", file=sys.stderr)
        return 2

    files = iter_input_files(args.input_dir, args.pattern, args.recursive)
    if not files:
        print(f"Nessun CSV trovato in {args.input_dir} con pattern {args.pattern}")
        return 0

    print(f"Input folder: {args.input_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"FIR offline: notch {args.notch:g} Hz + HP {args.hp:g} Hz, taps={args.taps}, fs={args.fs:g} Hz")
    print(f"Files: {len(files)}")

    failed = 0
    for path in files:
        out = output_path_for(path, args.input_dir, args.output_dir, args.suffix)
        try:
            count, skipped = filter_file(path, out, args.fs, args.hp, args.taps, args.notch, args.notch_bw, args.channels)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {exc}")
            continue

        message = f"[OK] {path.name} -> {out.name} ({count} colonne filtrate)"
        if skipped:
            message += f"; saltate: {', '.join(skipped)}"
        print(message)

    if failed:
        print(f"Completato con {failed} file falliti.", file=sys.stderr)
        return 1
    print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
