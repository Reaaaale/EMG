#!/usr/bin/env python3
"""Copy a label column from one CSV to another CSV.

Examples
--------
Single file pair, overwriting the target:

    python scripts/copy_labels.py \
      --source-csv dataset/raw_1khz_labeled/10_10_10_labeled.csv \
      --target-csv dataset/filtered_1khz_labeled/10_10_10_labeled_filtered.csv \
      --overwrite

Batch mode, matching *_labeled.csv to *_labeled_filtered.csv:

    python scripts/copy_labels.py \
      --all-labeled-to-filtered \
      --source-dir "dataset/raw_1khz_labeled/additional_trials" \
      --target-dir dataset/filtered_1khz_labeled \
      --overwrite

Batch mode, matching *_filtrato_labeled.csv to raw recordings:

    python scripts/copy_labels.py \
      --all-clean-to-raw \
      --source-dir dataset_clean \
      --target-dir dataset_raw \
      --output-dir dataset_raw_labeled \
      --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy labels between CSV files.")
    parser.add_argument("--source-csv", type=Path, default=None, help="CSV containing the label column.")
    parser.add_argument("--target-csv", type=Path, default=None, help="CSV that will receive the label column.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Default: target CSV with --overwrite.")
    parser.add_argument("--source-dir", type=Path, default=Path("dataset/raw_1khz_labeled"), help="Source folder for batch mode.")
    parser.add_argument("--target-dir", type=Path, default=Path("dataset/filtered_1khz_labeled"), help="Target folder for batch mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for --all-clean-to-raw. Default: target directory.",
    )
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--all-labeled-to-filtered", action="store_true", help="Batch copy *_labeled.csv -> *_labeled_filtered.csv.")
    parser.add_argument(
        "--all-clean-to-raw",
        action="store_true",
        help="Batch copy *_filtrato_labeled.csv labels to matching raw CSVs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target/output files.")
    return parser.parse_args()


def filtered_name_from_labeled(source_name: str) -> str:
    if not source_name.endswith("_labeled.csv"):
        raise ValueError(f"Source file does not match *_labeled.csv: {source_name}")
    return source_name.replace("_labeled.csv", "_labeled_filtered.csv")


def copy_labels_between_files(
    source_csv: Path,
    target_csv: Path,
    output: Path | None,
    label_col: str,
    overwrite: bool,
) -> None:
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    out_path = output if output is not None else target_csv
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")

    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)

    print(f"source shape: {source_df.shape}")
    print(f"target shape: {target_df.shape}")

    if len(source_df) != len(target_df):
        raise ValueError(
            f"Row count mismatch: source has {len(source_df)} rows, "
            f"target has {len(target_df)} rows."
        )
    if label_col not in source_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {source_csv}. "
            f"Available columns: {list(source_df.columns)}"
        )

    labeled_df = target_df.copy()
    labeled_df[label_col] = source_df[label_col].to_numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_path, index=False)

    counts = labeled_df[label_col].value_counts().sort_index().to_dict()
    print(f"label distribution: {counts}")
    print(f"saved: {out_path}")


def copy_all_labeled_to_filtered(source_dir: Path, target_dir: Path, label_col: str, overwrite: bool) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    sources = sorted(source_dir.glob("*_labeled.csv"))
    if not sources:
        raise FileNotFoundError(f"No *_labeled.csv files found in {source_dir}")

    copied = 0
    skipped = 0
    for source_csv in sources:
        target_csv = target_dir / filtered_name_from_labeled(source_csv.name)
        if not target_csv.exists():
            print(f"SKIP missing target: {target_csv}")
            skipped += 1
            continue

        print(f"\n=== {source_csv.name} -> {target_csv.name} ===")
        copy_labels_between_files(source_csv, target_csv, None, label_col, overwrite)
        copied += 1

    print(f"\nDone. copied={copied}, skipped={skipped}")


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as csv_file:
        return max(0, sum(1 for _ in csv_file) - 1)


def _normalized_name(path: Path) -> str:
    return path.name.lower().replace(" ", "_")


def copy_all_clean_to_raw(
    source_dir: Path,
    target_dir: Path,
    output_dir: Path | None,
    label_col: str,
    overwrite: bool,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    sources = sorted(source_dir.glob("*_filtrato_labeled.csv"))
    targets = sorted(
        path for path in target_dir.glob("*.csv")
        if not path.name.endswith("_labeled.csv")
    )
    if not sources:
        raise FileNotFoundError(f"No *_filtrato_labeled.csv files found in {source_dir}")
    if not targets:
        raise FileNotFoundError(f"No raw CSV files found in {target_dir}")

    output_dir = target_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = {path: _csv_row_count(path) for path in targets}
    unused_targets = set(targets)
    copied = 0

    for source_csv in sources:
        prefix = source_csv.name.removesuffix("_filtrato_labeled.csv")
        expected_name = f"{prefix}_raw.csv".lower()
        source_rows = _csv_row_count(source_csv)

        named_matches = [
            path for path in unused_targets
            if _normalized_name(path) == expected_name
        ]
        target_csv = named_matches[0] if named_matches else None

        if target_csv is None or target_rows[target_csv] != source_rows:
            row_matches = [
                path for path in unused_targets
                if target_rows[path] == source_rows
            ]
            if len(row_matches) != 1:
                raise ValueError(
                    f"Cannot uniquely match {source_csv.name} ({source_rows} rows): "
                    f"candidates={[path.name for path in row_matches]}"
                )
            target_csv = row_matches[0]
            print(
                f"ROW-COUNT MATCH: {source_csv.name} -> {target_csv.name} "
                f"({source_rows} rows)"
            )

        output_csv = output_dir / f"{prefix}_raw_labeled.csv"
        print(f"\n=== {source_csv.name} -> {target_csv.name} -> {output_csv.name} ===")
        copy_labels_between_files(
            source_csv,
            target_csv,
            output_csv,
            label_col,
            overwrite,
        )
        unused_targets.remove(target_csv)
        copied += 1

    print(f"\nDone. copied={copied}")


def main() -> None:
    args = parse_args()
    if args.all_clean_to_raw:
        copy_all_clean_to_raw(
            args.source_dir,
            args.target_dir,
            args.output_dir,
            args.label_col,
            args.overwrite,
        )
        return
    if args.all_labeled_to_filtered:
        copy_all_labeled_to_filtered(args.source_dir, args.target_dir, args.label_col, args.overwrite)
        return

    if args.source_csv is None or args.target_csv is None:
        raise ValueError("Use --source-csv and --target-csv, or pass --all-labeled-to-filtered.")

    copy_labels_between_files(args.source_csv, args.target_csv, args.output, args.label_col, args.overwrite)


if __name__ == "__main__":
    main()
