from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6 import QtWidgets

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_emg.processing import (
    apply_butterworth_offline_zero_phase,
    apply_butterworth_realtime_fixed_point,
    apply_butterworth_realtime_causal,
    apply_fir_offline_zero_phase,
    apply_fir_realtime_fixed_point,
    group_delay_samples,
    signal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtering package: raw CSVs, FIR-filtered CSVs, Butterworth-filtered CSVs and plots."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output package folder.")
    parser.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz.")
    parser.add_argument("--hp", type=float, default=5.0, help="High-pass cutoff in Hz.")
    parser.add_argument("--lp", type=float, default=None, help="Optional low-pass cutoff in Hz.")
    parser.add_argument("--fir-taps", type=int, default=51, help="FIR tap count for notch and high-pass.")
    parser.add_argument("--frac-bits", type=int, default=15, help="Fractional bits for realtime fixed-point FIR.")
    parser.add_argument("--butter-order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch center frequency in Hz.")
    parser.add_argument("--notch-bw", type=float, default=2.0, help="FIR notch transition bandwidth in Hz.")
    parser.add_argument("--notch-q", type=float, default=30.0, help="IIR notch Q for the Butterworth pipeline.")
    parser.add_argument("--pattern", default="*.csv", help="Input file glob pattern.")
    parser.add_argument("--recursive", action="store_true", help="Search input files recursively.")
    parser.add_argument("--plot-channel", default=None, help="Single channel to plot. Default: plot all filtered channels.")
    parser.add_argument("--no-png", action="store_true", help="Do not create per-channel PNG plots.")
    parser.add_argument("--html-report", default="report.html", help="Single interactive HTML report filename.")
    parser.add_argument("--html-max-points", type=int, default=6000, help="Maximum plotted points per channel in HTML.")
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


def iter_input_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def relative_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / input_path.relative_to(input_dir)


def clone_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(row) for row in rows]


def filter_rows(
    rows: list[dict[str, str]],
    channels: list[str],
    fs: float,
    hp: float,
    lp: float | None,
    fir_taps: int,
    fractional_bits: int,
    butter_order: int,
    notch: float,
    notch_bw: float,
    notch_q: float,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    fir_rows = clone_rows(rows)
    fir_realtime_rows = clone_rows(rows)
    butter_rows = clone_rows(rows)
    butter_realtime_rows = clone_rows(rows)
    butter_fixed_rows = clone_rows(rows)
    raw_data: dict[str, np.ndarray] = {}
    fir_data: dict[str, np.ndarray] = {}
    fir_realtime_data: dict[str, np.ndarray] = {}
    butter_data: dict[str, np.ndarray] = {}
    butter_realtime_data: dict[str, np.ndarray] = {}
    butter_fixed_data: dict[str, np.ndarray] = {}

    for channel in channels:
        x = np.array([float(row[channel]) for row in rows], dtype=float)
        y_fir = apply_fir_offline_zero_phase(
            x,
            fs=fs,
            hp_cutoff_hz=hp,
            numtaps=fir_taps,
            notch_hz=notch,
            notch_bw_hz=notch_bw,
        )
        y_fir_realtime = apply_fir_realtime_fixed_point(
            x,
            fs=fs,
            hp_cutoff_hz=hp,
            numtaps=fir_taps,
            fractional_bits=fractional_bits,
            notch_hz=notch,
            notch_bw_hz=notch_bw,
        )
        y_butter = apply_butterworth_offline_zero_phase(
            x,
            fs=fs,
            hp_cutoff_hz=hp,
            lp_cutoff_hz=lp,
            order=butter_order,
            notch_hz=notch,
            notch_q=notch_q,
        )
        y_butter_realtime = apply_butterworth_realtime_causal(
            x,
            fs=fs,
            hp_cutoff_hz=hp,
            lp_cutoff_hz=lp,
            order=butter_order,
            notch_hz=notch,
            notch_q=notch_q,
        )
        y_butter_fixed = apply_butterworth_realtime_fixed_point(
            x,
            fs=fs,
            hp_cutoff_hz=hp,
            lp_cutoff_hz=lp,
            order=butter_order,
            fractional_bits=fractional_bits,
            notch_hz=notch,
            notch_q=notch_q,
        )

        raw_data[channel] = x
        fir_data[channel] = y_fir
        fir_realtime_data[channel] = y_fir_realtime
        butter_data[channel] = y_butter
        butter_realtime_data[channel] = y_butter_realtime
        butter_fixed_data[channel] = y_butter_fixed

        for row, value in zip(fir_rows, y_fir):
            row[channel] = f"{value:.12g}"
        for row, value in zip(fir_realtime_rows, y_fir_realtime):
            row[channel] = f"{value:.12g}"
        for row, value in zip(butter_rows, y_butter):
            row[channel] = f"{value:.12g}"
        for row, value in zip(butter_realtime_rows, y_butter_realtime):
            row[channel] = f"{value:.12g}"
        for row, value in zip(butter_fixed_rows, y_butter_fixed):
            row[channel] = f"{value:.12g}"

    return (
        fir_rows,
        fir_realtime_rows,
        butter_rows,
        butter_realtime_rows,
        butter_fixed_rows,
        raw_data,
        fir_data,
        fir_realtime_data,
        butter_data,
        butter_realtime_data,
        butter_fixed_data,
    )


def plot_channel(
    path: Path,
    t: np.ndarray,
    channel: str,
    raw: np.ndarray,
    fir: np.ndarray,
    fir_realtime: np.ndarray,
    butter: np.ndarray,
    butter_realtime: np.ndarray,
    butter_fixed: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    window = pg.GraphicsLayoutWidget(show=False, title=f"{title} - {subtitle}")
    window.resize(1500, 850)

    p1 = window.addPlot(title=f"{title}<br>{subtitle}")
    p1.setLabel("bottom", "Time", units="s")
    p1.setLabel("left", "Amplitude", units="counts")
    p1.showGrid(x=True, y=True, alpha=0.25)
    p1.addLegend()
    p1.plot(t, raw, pen=pg.mkPen((150, 150, 150), width=1), name="raw")
    p1.plot(t, fir, pen=pg.mkPen((59, 130, 246), width=2), name="FIR: notch 50 Hz + HP 5 Hz")
    p1.plot(t, fir_realtime, pen=pg.mkPen((20, 150, 90), width=2), name="FIR realtime causal")
    p1.plot(t, butter, pen=pg.mkPen((220, 90, 70), width=2), name="Butterworth: notch 50 Hz + HP 5 Hz")
    p1.plot(t, butter_realtime, pen=pg.mkPen((245, 158, 11), width=2), name="Butterworth realtime causal")
    p1.plot(t, butter_fixed, pen=pg.mkPen((147, 51, 234), width=2), name="Butterworth realtime fixed-point")

    window.nextRow()
    p2 = window.addPlot(title=f"{channel}: Butterworth - FIR offline")
    p2.setLabel("bottom", "Time", units="s")
    p2.setLabel("left", "Difference", units="counts")
    p2.showGrid(x=True, y=True, alpha=0.25)
    p2.plot(t, butter - fir, pen=pg.mkPen((120, 80, 180), width=1.5))

    path.parent.mkdir(parents=True, exist_ok=True)
    exporter = pyqtgraph.exporters.ImageExporter(window.scene())
    exporter.parameters()["width"] = 1500
    exporter.export(str(path))
    window.close()


def decimation_indices(series: list[np.ndarray], max_points: int) -> np.ndarray:
    n = len(series[0])
    if n <= max_points:
        return np.arange(n)

    picks_per_bucket = max(2, 2 * len(series))
    bucket_count = max(1, max_points // picks_per_bucket)
    edges = np.linspace(0, n, bucket_count + 1, dtype=int)
    indices: set[int] = {0, n - 1}

    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        for values in series:
            segment = values[start:stop]
            if len(segment) == 0:
                continue
            indices.add(start + int(np.argmin(segment)))
            indices.add(start + int(np.argmax(segment)))

    return np.array(sorted(indices), dtype=int)


def rounded(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(values.astype(float), digits).tolist()


def spectrum_db(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    centered = x.astype(float) - np.mean(x)
    if len(centered) < 2:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(len(centered))
    scale = np.sum(window) / 2
    freq = np.fft.rfftfreq(len(centered), d=1 / fs)
    mag = np.abs(np.fft.rfft(centered * window)) / max(scale, 1e-12)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    return freq, mag_db


def decimate_frequency(freq: np.ndarray, series: list[np.ndarray], max_points: int) -> np.ndarray:
    n = len(freq)
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def report_entry(
    path: Path,
    rows: list[dict[str, str]],
    channels: list[str],
    raw_data: dict[str, np.ndarray],
    fir_data: dict[str, np.ndarray],
    fir_realtime_data: dict[str, np.ndarray],
    butter_data: dict[str, np.ndarray],
    butter_realtime_data: dict[str, np.ndarray],
    butter_fixed_data: dict[str, np.ndarray],
    fs: float,
    max_points: int,
) -> dict[str, object]:
    t = np.arange(len(rows), dtype=float) / fs
    channel_payload: dict[str, object] = {}
    for channel in channels:
        idx = decimation_indices(
            [
                raw_data[channel],
                fir_data[channel],
                fir_realtime_data[channel],
                butter_data[channel],
                butter_realtime_data[channel],
                butter_fixed_data[channel],
            ],
            max_points,
        )
        raw_freq, raw_spec = spectrum_db(raw_data[channel], fs)
        fir_freq, fir_spec = spectrum_db(fir_data[channel], fs)
        fir_realtime_freq, fir_realtime_spec = spectrum_db(fir_realtime_data[channel], fs)
        butter_freq, butter_spec = spectrum_db(butter_data[channel], fs)
        butter_realtime_freq, butter_realtime_spec = spectrum_db(butter_realtime_data[channel], fs)
        butter_fixed_freq, butter_fixed_spec = spectrum_db(butter_fixed_data[channel], fs)
        fidx = decimate_frequency(
            raw_freq,
            [raw_spec, fir_spec, fir_realtime_spec, butter_spec, butter_realtime_spec, butter_fixed_spec],
            max_points,
        )
        channel_payload[channel] = {
            "t": rounded(t[idx], 5),
            "raw": rounded(raw_data[channel][idx], 4),
            "fir": rounded(fir_data[channel][idx], 4),
            "firRealtime": rounded(fir_realtime_data[channel][idx], 4),
            "butter": rounded(butter_data[channel][idx], 4),
            "butterRealtime": rounded(butter_realtime_data[channel][idx], 4),
            "butterFixed": rounded(butter_fixed_data[channel][idx], 4),
            "freq": rounded(raw_freq[fidx], 4),
            "rawSpec": rounded(raw_spec[fidx], 3),
            "firSpec": rounded(fir_spec[fidx], 3),
            "firRealtimeSpec": rounded(fir_realtime_spec[fidx], 3),
            "butterSpec": rounded(butter_spec[fidx], 3),
            "butterRealtimeSpec": rounded(butter_realtime_spec[fidx], 3),
            "butterFixedSpec": rounded(butter_fixed_spec[fidx], 3),
        }
    return {"name": path.name, "samples": len(rows), "channels": channel_payload}


def write_html_report(path: Path, payload: dict[str, object]) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMG Filter Report</title>
<style>
  :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ margin: 0; background: #f6f7f9; color: #15171a; }}
  header {{ padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #dfe3e8; position: sticky; top: 0; z-index: 2; }}
  h1 {{ margin: 0 0 10px; font-size: 20px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }}
  label {{ font-size: 13px; color: #3f4650; }}
  select, input {{ font: inherit; padding: 6px 8px; border: 1px solid #c9d0d8; border-radius: 6px; background: #fff; }}
  main {{ padding: 18px 24px 28px; }}
  .panel {{ background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 14px; }}
  canvas {{ width: 100%; height: 560px; display: block; }}
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 0; font-size: 13px; }}
  .swatch {{ display: inline-block; width: 20px; height: 3px; vertical-align: middle; margin-right: 6px; }}
  .meta {{ margin-top: 10px; color: #606875; font-size: 13px; line-height: 1.45; }}
</style>
</head>
<body>
<header>
  <h1>EMG Filter Report</h1>
  <div class="controls">
    <label>File <select id="file"></select></label>
    <label>Canale <select id="channel"></select></label>
    <label><input id="raw" type="checkbox" checked> Raw</label>
    <label><input id="fir" type="checkbox" checked> FIR</label>
    <label><input id="firRealtime" type="checkbox" checked> FIR realtime</label>
    <label><input id="butter" type="checkbox" checked> Butterworth</label>
    <label><input id="butterRealtime" type="checkbox" checked> Butter realtime</label>
    <label><input id="butterFixed" type="checkbox" checked> Butter fixed</label>
    <label><input id="diff" type="checkbox"> Differenza Butter - FIR</label>
    <label>Start [s] <input id="start" type="number" min="0" step="0.1" value="0"></label>
    <label>End [s] <input id="end" type="number" min="0" step="0.1" value="0"></label>
    <label>F max [Hz] <input id="fmax" type="number" min="1" step="1" value="120"></label>
    <button id="full">Tutto</button>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Tempo</h2>
    <canvas id="plot" width="1600" height="720"></canvas>
    <div class="legend">
      <span><span class="swatch" style="background:#999"></span>raw</span>
      <span><span class="swatch" style="background:#2563eb"></span>FIR: notch 50 Hz + HP</span>
      <span><span class="swatch" style="background:#14965a"></span>FIR realtime causale</span>
      <span><span class="swatch" style="background:#dc5a46"></span>Butterworth: notch 50 Hz + HP</span>
      <span><span class="swatch" style="background:#f59e0b"></span>Butterworth realtime causale</span>
      <span><span class="swatch" style="background:#9333ea"></span>Butterworth realtime fixed-point</span>
      <span><span class="swatch" style="background:#7c3aed"></span>Butter - FIR</span>
    </div>
    <div id="meta" class="meta"></div>
  </section>
  <section class="panel" style="margin-top:16px">
    <h2>Fourier</h2>
    <canvas id="spectrum" width="1600" height="560"></canvas>
    <div class="meta">Spettro in dB con finestra Hann. Usa F max per zoomare attorno ai 50 Hz.</div>
  </section>
</main>
<script>
const REPORT = {data};
const $ = id => document.getElementById(id);
const fileSel = $("file"), chSel = $("channel"), canvas = $("plot"), ctx = canvas.getContext("2d");
const spectrumCanvas = $("spectrum"), sctx = spectrumCanvas.getContext("2d");

function fillFiles() {{
  REPORT.files.forEach((f, i) => {{
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = f.name;
    fileSel.appendChild(opt);
  }});
}}

function fillChannels() {{
  chSel.innerHTML = "";
  const file = REPORT.files[Number(fileSel.value)];
  Object.keys(file.channels).forEach(ch => {{
    const opt = document.createElement("option");
    opt.value = ch;
    opt.textContent = ch;
    chSel.appendChild(opt);
  }});
  setFullRange();
}}

function setFullRange() {{
  const d = currentData();
  const end = d.t[d.t.length - 1] || 0;
  $("start").value = "0";
  $("end").value = end.toFixed(2);
}}

function currentData() {{
  return REPORT.files[Number(fileSel.value)].channels[chSel.value];
}}

function visibleSeries(d, start, end) {{
  const out = [];
  if ($("raw").checked) out.push(["raw", d.raw, "#999", 1.2]);
  if ($("fir").checked) out.push(["fir", d.fir, "#2563eb", 1.8]);
  if ($("firRealtime").checked) out.push(["fir realtime", d.firRealtime, "#14965a", 1.8]);
  if ($("butter").checked) out.push(["butter", d.butter, "#dc5a46", 1.8]);
  if ($("butterRealtime").checked) out.push(["butter realtime", d.butterRealtime, "#f59e0b", 1.8]);
  if ($("butterFixed").checked) out.push(["butter fixed", d.butterFixed, "#9333ea", 1.8]);
  if ($("diff").checked) out.push(["butter - fir", d.butter.map((v, i) => v - d.fir[i]), "#7c3aed", 1.5]);
  return out.map(([name, y, color, width]) => {{
    const pts = [];
    for (let i = 0; i < d.t.length; i++) {{
      if (d.t[i] >= start && d.t[i] <= end) pts.push([d.t[i], y[i]]);
    }}
    return {{name, pts, color, width}};
  }}).filter(s => s.pts.length);
}}

function draw() {{
  const d = currentData();
  const start = Number($("start").value);
  const end = Number($("end").value);
  const series = visibleSeries(d, start, end);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pad = {{l: 70, r: 25, t: 24, b: 52}};
  const w = canvas.width - pad.l - pad.r;
  const h = canvas.height - pad.t - pad.b;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let ymin = Infinity, ymax = -Infinity;
  series.forEach(s => s.pts.forEach(p => {{ ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }}));
  if (!isFinite(ymin) || ymin === ymax) {{ ymin = -1; ymax = 1; }}
  const margin = (ymax - ymin) * 0.08 || 1;
  ymin -= margin; ymax += margin;
  const xmin = start, xmax = end > start ? end : start + 1;
  const xmap = x => pad.l + ((x - xmin) / (xmax - xmin)) * w;
  const ymap = y => pad.t + (1 - (y - ymin) / (ymax - ymin)) * h;

  ctx.strokeStyle = "#e3e7ed"; ctx.lineWidth = 1;
  ctx.fillStyle = "#68717d"; ctx.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let i = 0; i <= 10; i++) {{
    const x = pad.l + i * w / 10;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + h); ctx.stroke();
    const label = (xmin + i * (xmax - xmin) / 10).toFixed(1);
    ctx.fillText(label, x - 12, pad.t + h + 28);
  }}
  for (let i = 0; i <= 8; i++) {{
    const y = pad.t + i * h / 8;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    const label = (ymax - i * (ymax - ymin) / 8).toPrecision(4);
    ctx.fillText(label, 8, y + 4);
  }}

  ctx.strokeStyle = "#20242a"; ctx.lineWidth = 1.2;
  ctx.strokeRect(pad.l, pad.t, w, h);

  series.forEach(s => {{
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
    ctx.beginPath();
    s.pts.forEach((p, i) => {{
      const x = xmap(p[0]), y = ymap(p[1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});

  $("meta").textContent = `${{REPORT.meta.fs}} Hz | ${{REPORT.meta.fir}} | ${{REPORT.meta.firRealtime}} | ${{REPORT.meta.butterworth}} | ${{REPORT.meta.butterworthRealtime}} | ${{REPORT.meta.butterworthFixed}} | campioni originali: ${{REPORT.files[Number(fileSel.value)].samples}}`;
}}

function spectralSeries(d, fmax) {{
  const out = [];
  if ($("raw").checked) out.push(["raw", d.rawSpec, "#999", 1.2]);
  if ($("fir").checked) out.push(["fir", d.firSpec, "#2563eb", 1.8]);
  if ($("firRealtime").checked) out.push(["fir realtime", d.firRealtimeSpec, "#14965a", 1.8]);
  if ($("butter").checked) out.push(["butter", d.butterSpec, "#dc5a46", 1.8]);
  if ($("butterRealtime").checked) out.push(["butter realtime", d.butterRealtimeSpec, "#f59e0b", 1.8]);
  if ($("butterFixed").checked) out.push(["butter fixed", d.butterFixedSpec, "#9333ea", 1.8]);
  return out.map(([name, y, color, width]) => {{
    const pts = [];
    for (let i = 0; i < d.freq.length; i++) {{
      if (d.freq[i] <= fmax) pts.push([d.freq[i], y[i]]);
    }}
    return {{name, pts, color, width}};
  }}).filter(s => s.pts.length);
}}

function drawSpectrum() {{
  const d = currentData();
  const fmax = Number($("fmax").value) || Number(REPORT.meta.fs) / 2;
  const series = spectralSeries(d, fmax);
  sctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

  const pad = {{l: 70, r: 25, t: 24, b: 52}};
  const w = spectrumCanvas.width - pad.l - pad.r;
  const h = spectrumCanvas.height - pad.t - pad.b;
  sctx.fillStyle = "#fff";
  sctx.fillRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

  let ymin = Infinity, ymax = -Infinity;
  series.forEach(s => s.pts.forEach(p => {{ ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }}));
  if (!isFinite(ymin) || ymin === ymax) {{ ymin = -120; ymax = 0; }}
  const margin = (ymax - ymin) * 0.08 || 1;
  ymin -= margin; ymax += margin;
  const xmin = 0, xmax = Math.max(1, fmax);
  const xmap = x => pad.l + ((x - xmin) / (xmax - xmin)) * w;
  const ymap = y => pad.t + (1 - (y - ymin) / (ymax - ymin)) * h;

  sctx.strokeStyle = "#e3e7ed"; sctx.lineWidth = 1;
  sctx.fillStyle = "#68717d"; sctx.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let i = 0; i <= 10; i++) {{
    const x = pad.l + i * w / 10;
    sctx.beginPath(); sctx.moveTo(x, pad.t); sctx.lineTo(x, pad.t + h); sctx.stroke();
    const label = (xmin + i * (xmax - xmin) / 10).toFixed(0);
    sctx.fillText(label, x - 10, pad.t + h + 28);
  }}
  for (let i = 0; i <= 8; i++) {{
    const y = pad.t + i * h / 8;
    sctx.beginPath(); sctx.moveTo(pad.l, y); sctx.lineTo(pad.l + w, y); sctx.stroke();
    const label = (ymax - i * (ymax - ymin) / 8).toFixed(1);
    sctx.fillText(label, 8, y + 4);
  }}

  const x50 = xmap(50);
  if (x50 >= pad.l && x50 <= pad.l + w) {{
    sctx.strokeStyle = "#111"; sctx.setLineDash([6, 5]);
    sctx.beginPath(); sctx.moveTo(x50, pad.t); sctx.lineTo(x50, pad.t + h); sctx.stroke();
    sctx.setLineDash([]);
    sctx.fillStyle = "#111"; sctx.fillText("50 Hz", x50 + 6, pad.t + 16);
  }}

  sctx.strokeStyle = "#20242a"; sctx.lineWidth = 1.2;
  sctx.strokeRect(pad.l, pad.t, w, h);

  series.forEach(s => {{
    sctx.strokeStyle = s.color; sctx.lineWidth = s.width;
    sctx.beginPath();
    s.pts.forEach((p, i) => {{
      const x = xmap(p[0]), y = ymap(p[1]);
      if (i === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
    }});
    sctx.stroke();
  }});
}}

function redrawAll() {{
  draw();
  drawSpectrum();
}}

fillFiles();
fillChannels();
[fileSel, chSel, $("raw"), $("fir"), $("firRealtime"), $("butter"), $("butterRealtime"), $("butterFixed"), $("diff"), $("start"), $("end"), $("fmax")].forEach(el => {{
  el.addEventListener("change", () => {{ if (el === fileSel) fillChannels(); redrawAll(); }});
  el.addEventListener("input", redrawAll);
}});
$("full").addEventListener("click", () => {{ setFullRange(); redrawAll(); }});
redrawAll();
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def write_report(path: Path, args: argparse.Namespace, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lp = "disabled" if args.lp is None else f"{args.lp:g} Hz"
    with path.open("w") as f:
        f.write("Batch filtering package\n")
        f.write("=======================\n\n")
        f.write(f"Input folder: {args.input_dir}\n")
        f.write(f"Files: {len(files)}\n")
        f.write(f"Fs: {args.fs:g} Hz\n")
        f.write(f"Channels: {'auto CH<number>' if args.channels is None else ', '.join(args.channels)}\n\n")
        f.write("Output folders:\n")
        f.write("- raw/: copie identiche dei CSV originali, senza filtro\n")
        f.write("- fir/: CSV filtrati con FIR offline zero-phase\n")
        f.write("- fir_realtime/: CSV filtrati con FIR causale fixed-point, ritardo incluso\n")
        f.write("- butterworth/: CSV filtrati con Butterworth/IIR offline zero-phase\n")
        f.write("- butterworth_realtime/: CSV filtrati con Butterworth/IIR causale, fase/ritardo realtime inclusi\n")
        f.write("- butterworth_fixed/: CSV filtrati con Butterworth/IIR causale fixed-point\n")
        f.write("- plots/: PNG raw vs FIR offline/realtime vs Butterworth offline/realtime\n\n")
        f.write(
            f"FIR offline: notch FIR {args.notch:g} Hz, bandwidth {args.notch_bw:g} Hz "
            f"+ high-pass FIR {args.hp:g} Hz, taps={args.fir_taps}, applicati con filtfilt\n"
        )
        delay = 2 * group_delay_samples(args.fir_taps)
        f.write(
            f"FIR realtime: stesso notch FIR + high-pass FIR, causale sample-by-sample, "
            f"Q*.{args.frac_bits}, ritardo teorico {delay} campioni = {delay / args.fs:.4f} s\n"
        )
        f.write(
            f"Butterworth offline: notch IIR {args.notch:g} Hz Q={args.notch_q:g} "
            f"+ high-pass Butterworth {args.hp:g} Hz ordine {args.butter_order} "
            f"+ low-pass {lp}, applicati con filtfilt\n"
        )
        f.write(
            f"Butterworth realtime: stesso filtro IIR in sezioni SOS/biquad, causale con sosfilt, "
            f"fase e transitorio realtime inclusi\n"
        )
        f.write(
            f"Butterworth fixed-point: stesso filtro IIR/SOS causale, coefficienti Q*.{args.frac_bits}, "
            f"stati interi e accumulo intero\n"
        )


def process_file(path: Path, args: argparse.Namespace, app: QtWidgets.QApplication) -> tuple[int, int, dict[str, object]]:
    del app
    fieldnames, rows = read_csv(path)
    if not rows:
        raise ValueError("file CSV vuoto")

    channels = args.channels if args.channels is not None else default_channel_names(fieldnames)
    if args.plot_channel is not None:
        channels_to_plot = [args.plot_channel]
    else:
        channels_to_plot = channels
    missing = [name for name in channels + channels_to_plot if name not in fieldnames]
    if missing:
        raise ValueError(f"colonne non trovate: {', '.join(sorted(set(missing)))}")

    (
        fir_rows,
        fir_realtime_rows,
        butter_rows,
        butter_realtime_rows,
        butter_fixed_rows,
        raw_data,
        fir_data,
        fir_realtime_data,
        butter_data,
        butter_realtime_data,
        butter_fixed_data,
    ) = filter_rows(
        rows,
        channels,
        fs=args.fs,
        hp=args.hp,
        lp=args.lp,
        fir_taps=args.fir_taps,
        fractional_bits=args.frac_bits,
        butter_order=args.butter_order,
        notch=args.notch,
        notch_bw=args.notch_bw,
        notch_q=args.notch_q,
    )

    raw_out = relative_output_path(path, args.input_dir, args.output_dir / "raw")
    fir_out = relative_output_path(path, args.input_dir, args.output_dir / "fir")
    fir_realtime_out = relative_output_path(path, args.input_dir, args.output_dir / "fir_realtime")
    butter_out = relative_output_path(path, args.input_dir, args.output_dir / "butterworth")
    butter_realtime_out = relative_output_path(path, args.input_dir, args.output_dir / "butterworth_realtime")
    butter_fixed_out = relative_output_path(path, args.input_dir, args.output_dir / "butterworth_fixed")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, raw_out)
    write_csv(fir_out, fieldnames, fir_rows)
    write_csv(fir_realtime_out, fieldnames, fir_realtime_rows)
    write_csv(butter_out, fieldnames, butter_rows)
    write_csv(butter_realtime_out, fieldnames, butter_realtime_rows)
    write_csv(butter_fixed_out, fieldnames, butter_fixed_rows)

    t = np.arange(len(rows)) / args.fs
    plot_count = 0
    if not args.no_png:
        for channel in channels_to_plot:
            if channel not in raw_data:
                continue
            lp_label = "LP off" if args.lp is None else f"LP {args.lp:g} Hz"
            subtitle = (
                f"FIR: notch {args.notch:g} Hz + HP {args.hp:g} Hz, taps={args.fir_taps} | "
                f"Butterworth: notch {args.notch:g} Hz Q={args.notch_q:g} + HP {args.hp:g} Hz, "
                f"ordine {args.butter_order}, {lp_label}"
            )
            plot_out = relative_output_path(
                path.with_name(f"{path.stem}_{channel}.png"),
                args.input_dir,
                args.output_dir / "plots",
            )
            plot_channel(
                plot_out,
                t,
                channel,
                raw_data[channel],
                fir_data[channel],
                fir_realtime_data[channel],
                butter_data[channel],
                butter_realtime_data[channel],
                butter_fixed_data[channel],
                f"{path.name} {channel}",
                subtitle,
            )
            plot_count += 1

    html_entry = report_entry(
        path,
        rows,
        channels,
        raw_data,
        fir_data,
        fir_realtime_data,
        butter_data,
        butter_realtime_data,
        butter_fixed_data,
        fs=args.fs,
        max_points=args.html_max_points,
    )

    return len(channels), plot_count, html_entry


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

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    write_report(args.output_dir / "README_filtering.txt", args, files)
    print(f"Output package: {args.output_dir}")
    print(f"Files: {len(files)}")

    failed = 0
    html_entries: list[dict[str, object]] = []
    for path in files:
        try:
            channel_count, plot_count, html_entry = process_file(path, args, app)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {exc}")
            continue
        html_entries.append(html_entry)
        print(
            f"[OK] {path.name}: raw + FIR offline + FIR realtime + "
            f"Butterworth offline + Butterworth realtime + Butterworth fixed, "
            f"{channel_count} canali, {plot_count} plot"
        )

    if html_entries:
        lp = "LP off" if args.lp is None else f"LP {args.lp:g} Hz"
        delay = 2 * group_delay_samples(args.fir_taps)
        html_payload = {
            "meta": {
                "fs": f"{args.fs:g}",
                "fir": f"FIR notch {args.notch:g} Hz + HP {args.hp:g} Hz, taps={args.fir_taps}, filtfilt",
                "firRealtime": (
                    f"FIR realtime causale Q*.{args.frac_bits}, "
                    f"ritardo {delay} campioni = {delay / args.fs:.4f} s"
                ),
                "butterworth": (
                    f"IIR notch {args.notch:g} Hz Q={args.notch_q:g} + "
                    f"Butterworth HP {args.hp:g} Hz ordine {args.butter_order}, {lp}, filtfilt"
                ),
                "butterworthRealtime": (
                    f"IIR notch {args.notch:g} Hz Q={args.notch_q:g} + "
                    f"Butterworth HP {args.hp:g} Hz ordine {args.butter_order}, {lp}, causale sosfilt"
                ),
                "butterworthFixed": (
                    f"IIR/SOS Butterworth realtime fixed-point Q*.{args.frac_bits}, "
                    f"coefficienti e stati interi"
                ),
            },
            "files": html_entries,
        }
        report_path = args.output_dir / args.html_report
        write_html_report(report_path, html_payload)
        print(f"HTML report: {report_path}")

    if failed:
        print(f"Completato con {failed} file falliti.", file=sys.stderr)
        return 1
    print("Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
