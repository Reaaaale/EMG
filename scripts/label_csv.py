from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CSV signal labeler.")
    parser.add_argument("csv_file", nargs="?", type=Path, help="CSV file to open.")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency in Hz.")
    parser.add_argument("--channel", default="CH1", help="Initial channel.")
    parser.add_argument("--label", default="active", help="Label assigned to selected regions.")
    parser.add_argument("--background-label", default="rest", help="Label assigned outside selected regions.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    return parser.parse_args()


def numeric_channel_names(fieldnames: list[str]) -> list[str]:
    channels: list[str] = []
    for name in fieldnames:
        if name.upper().startswith("CH"):
            channels.append(name)
    return channels


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV senza header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    data = np.zeros((len(rows), len(fieldnames)), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(fieldnames):
            value = row.get(name, "")
            try:
                data[row_idx, col_idx] = float(value)
            except ValueError:
                data[row_idx, col_idx] = np.nan
    return fieldnames, data


def write_labeled_csv(path: Path, fieldnames: list[str], data: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*fieldnames, "label"])
        for row, label in zip(data, labels):
            writer.writerow([*row, label])


@dataclass
class LabeledRegion:
    start_s: float
    end_s: float
    label: str
    item: pg.LinearRegionItem


class CsvLabeler(QtWidgets.QMainWindow):
    def __init__(
        self,
        csv_file: Path | None,
        fs: float,
        initial_channel: str,
        selected_label: str,
        background_label: str,
        output_path: Path | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("EMG CSV Labeler")
        self.resize(1350, 760)

        self.csv_path = csv_file
        self.output_path = output_path
        self.fieldnames: list[str] = []
        self.data = np.empty((0, 0), dtype=float)
        self.channels: list[str] = []
        self.regions: list[LabeledRegion] = []
        self.initial_channel = initial_channel

        self._build_ui(fs, selected_label, background_label)
        if csv_file is not None:
            self.load_csv(csv_file)

    def _build_ui(self, fs: float, selected_label: str, background_label: str) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QGridLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)

        controls = QtWidgets.QFrame()
        controls.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        controls_layout = QtWidgets.QGridLayout(controls)

        self.file_label = QtWidgets.QLabel("Nessun file")
        self.load_button = QtWidgets.QPushButton("Apri CSV")
        self.channel_combo = QtWidgets.QComboBox()
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(1.0, 100000.0)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setValue(fs)
        self.fs_spin.setSuffix(" Hz")

        self.selected_label_edit = QtWidgets.QLineEdit(selected_label)
        self.background_label_edit = QtWidgets.QLineEdit(background_label)
        self.add_button = QtWidgets.QPushButton("Aggiungi selezione")
        self.add_button.setToolTip("Invio: aggiunge la selezione corrente")
        self.edit_button = QtWidgets.QPushButton("Modifica selezione")
        self.update_button = QtWidgets.QPushButton("Aggiorna selezione")
        self.remove_button = QtWidgets.QPushButton("Rimuovi selezione")
        self.clear_button = QtWidgets.QPushButton("Pulisci")
        self.save_button = QtWidgets.QPushButton("Salva CSV etichettato")

        self.keyboard_help = QtWidgets.QLabel(
            "Invio aggiunge | ←/→: 10 campioni | Shift: 1 | Ctrl: 100"
        )

        controls_layout.addWidget(self.load_button, 0, 0)
        controls_layout.addWidget(self.file_label, 0, 1, 1, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Canale"), 1, 0)
        controls_layout.addWidget(self.channel_combo, 1, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Fs"), 1, 2)
        controls_layout.addWidget(self.fs_spin, 1, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Label selezione"), 2, 0)
        controls_layout.addWidget(self.selected_label_edit, 2, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Label resto"), 2, 2)
        controls_layout.addWidget(self.background_label_edit, 2, 3)
        controls_layout.addWidget(self.add_button, 3, 0)
        controls_layout.addWidget(self.edit_button, 3, 1)
        controls_layout.addWidget(self.update_button, 3, 2)
        controls_layout.addWidget(self.remove_button, 3, 3)
        controls_layout.addWidget(self.clear_button, 3, 4)
        controls_layout.addWidget(self.save_button, 3, 5)
        controls_layout.addWidget(self.keyboard_help, 4, 0, 1, 6)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time [s]")
        self.plot.setLabel("left", "Amplitude [counts]")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.addLegend()
        self.raw_curve = self.plot.plot([], [], pen=pg.mkPen("#8a8f98", width=1), name="raw")

        self.selection = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self.selection.setBrush(QtGui.QColor(49, 130, 206, 45))
        self.selection.setHoverBrush(QtGui.QColor(49, 130, 206, 80))
        self.plot.addItem(self.selection)

        self.region_list = QtWidgets.QListWidget()
        self.region_list.setMinimumWidth(330)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(120)

        layout.addWidget(controls, 0, 0, 1, 2)
        layout.addWidget(self.plot, 1, 0)
        layout.addWidget(self.region_list, 1, 1)
        layout.addWidget(self.status, 2, 0, 1, 2)
        layout.setColumnStretch(0, 1)

        self.load_button.clicked.connect(self.choose_csv)
        self.channel_combo.currentIndexChanged.connect(self.update_plot)
        self.fs_spin.valueChanged.connect(self.update_plot)
        self.add_button.clicked.connect(self.add_region)
        self.edit_button.clicked.connect(self.edit_selected_region)
        self.update_button.clicked.connect(self.update_selected_region)
        self.remove_button.clicked.connect(self.remove_selected_region)
        self.clear_button.clicked.connect(self.clear_regions)
        self.save_button.clicked.connect(self.save_labels)

        self.add_shortcuts = [
            QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return), self),
            QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Enter), self),
        ]
        for shortcut in self.add_shortcuts:
            shortcut.activated.connect(self.add_region)

        self.move_left_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        )
        self.move_right_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self
        )
        self.move_left_shortcut.activated.connect(lambda: self.move_selection(-1))
        self.move_right_shortcut.activated.connect(lambda: self.move_selection(1))

        self.move_fine_left_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Shift+Left"), self
        )
        self.move_fine_right_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Shift+Right"), self
        )
        self.move_fast_left_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Left"), self
        )
        self.move_fast_right_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Right"), self
        )
        self.move_fine_left_shortcut.activated.connect(lambda: self.move_selection(-1, 1))
        self.move_fine_right_shortcut.activated.connect(lambda: self.move_selection(1, 1))
        self.move_fast_left_shortcut.activated.connect(lambda: self.move_selection(-1, 100))
        self.move_fast_right_shortcut.activated.connect(lambda: self.move_selection(1, 100))

    def log(self, text: str) -> None:
        self.status.appendPlainText(text)

    def choose_csv(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Apri CSV",
            str(Path.cwd()),
            "CSV (*.csv);;Tutti i file (*)",
        )
        if filename:
            self.load_csv(Path(filename))

    def load_csv(self, path: Path) -> None:
        try:
            fieldnames, data = read_numeric_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Errore", f"Non riesco a leggere il CSV:\n{exc}")
            return

        self.csv_path = path
        self.fieldnames = fieldnames
        self.data = data
        self.channels = numeric_channel_names(fieldnames) or fieldnames
        self.file_label.setText(path.name)

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(self.channels)
        if self.initial_channel in self.channels:
            self.channel_combo.setCurrentText(self.initial_channel)
        self.channel_combo.blockSignals(False)

        self.update_plot()
        self.log(f"Caricato {path}: {data.shape[0]} campioni, {len(fieldnames)} colonne.")
        if self.regions:
            self.log(f"Mantenute {len(self.regions)} selezioni precedenti: spostale per adattarle al nuovo file.")

    def current_time_axis(self) -> np.ndarray:
        if self.data.size == 0:
            return np.array([], dtype=float)
        return np.arange(self.data.shape[0], dtype=float) / self.fs_spin.value()

    def update_plot(self) -> None:
        if self.data.size == 0 or not self.channels:
            return

        channel = self.channel_combo.currentText()
        if not channel:
            return

        col_idx = self.fieldnames.index(channel)
        x = self.current_time_axis()
        y = self.data[:, col_idx]
        self.raw_curve.setData(x, y)
        self.plot.setTitle(f"{self.csv_path.name if self.csv_path else 'CSV'} - {channel}")

        if x.size:
            duration = float(x[-1])
            self.plot.setXRange(0.0, duration, padding=0)
            start = max(0.0, duration * 0.05)
            end = max(start + 1.0 / self.fs_spin.value(), duration * 0.15)
            self.selection.setRegion((start, min(end, duration)))

            finite = y[np.isfinite(y)]
            if finite.size:
                low, high = np.percentile(finite, [1, 99])
                if low == high:
                    low -= 1.0
                    high += 1.0
                self.plot.setYRange(float(low), float(high), padding=0.12)

    def add_region(self) -> None:
        if self.data.size == 0:
            return

        start_s, end_s = self.selection.getRegion()
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        if end_s <= start_s:
            return

        duration = self.data.shape[0] / self.fs_spin.value()
        start_s = max(0.0, min(float(start_s), duration))
        end_s = max(0.0, min(float(end_s), duration))
        label = self.selected_label_edit.text().strip() or "active"

        item = pg.LinearRegionItem(values=(start_s, end_s), movable=False)
        item.setBrush(QtGui.QColor(34, 197, 94, 55))
        item.setHoverBrush(QtGui.QColor(34, 197, 94, 85))
        self.plot.addItem(item)

        region = LabeledRegion(start_s=start_s, end_s=end_s, label=label, item=item)
        self.regions.append(region)
        self.refresh_region_list()

    def move_selection(self, direction: int, sample_count: int = 10) -> None:
        """Move the blue selection by a number of samples without resizing it."""
        if self.data.size == 0 or direction == 0:
            return

        start_s, end_s = sorted(self.selection.getRegion())
        width_s = end_s - start_s
        if width_s <= 0:
            width_s = 1.0 / self.fs_spin.value()

        duration_s = self.data.shape[0] / self.fs_spin.value()
        width_s = min(width_s, duration_s)
        step_s = max(1, sample_count) / self.fs_spin.value()
        new_start_s = start_s + (step_s if direction > 0 else -step_s)
        new_start_s = max(0.0, min(new_start_s, duration_s - width_s))
        self.selection.setRegion((new_start_s, new_start_s + width_s))

    def edit_selected_region(self) -> None:
        row = self.region_list.currentRow()
        if row < 0 or row >= len(self.regions):
            self.log("Seleziona prima una regione dalla lista da modificare.")
            return
        region = self.regions[row]
        self.selection.setRegion((region.start_s, region.end_s))
        self.selected_label_edit.setText(region.label)
        self.log(f"Regione {row + 1} caricata nella selezione blu: spostala e premi 'Aggiorna selezione'.")

    def update_selected_region(self) -> None:
        row = self.region_list.currentRow()
        if row < 0 or row >= len(self.regions):
            self.log("Seleziona prima una regione dalla lista da aggiornare.")
            return

        region = self.regions[row]
        start_s, end_s = self.selection.getRegion()
        if end_s < start_s:
            start_s, end_s = end_s, start_s

        duration = self.data.shape[0] / self.fs_spin.value()
        region.start_s = max(0.0, min(float(start_s), duration))
        region.end_s = max(0.0, min(float(end_s), duration))
        region.label = self.selected_label_edit.text().strip() or region.label
        region.item.setRegion((region.start_s, region.end_s))
        self.refresh_region_list()
        self.region_list.setCurrentRow(row)
        self.log(f"Regione {row + 1} aggiornata.")

    def refresh_region_list(self) -> None:
        self.region_list.clear()
        for idx, region in enumerate(self.regions, start=1):
            self.region_list.addItem(
                f"{idx:02d} | {region.label} | {region.start_s:.3f}s - {region.end_s:.3f}s"
            )

    def remove_selected_region(self) -> None:
        row = self.region_list.currentRow()
        if row < 0 or row >= len(self.regions):
            return
        region = self.regions.pop(row)
        self.plot.removeItem(region.item)
        self.refresh_region_list()

    def clear_regions(self) -> None:
        for region in self.regions:
            self.plot.removeItem(region.item)
        self.regions.clear()
        self.region_list.clear()

    def labels_for_samples(self) -> np.ndarray:
        labels = np.full(self.data.shape[0], self.background_label_edit.text().strip() or "rest", dtype=object)
        fs = self.fs_spin.value()
        for region in self.regions:
            start_idx = int(np.floor(region.start_s * fs))
            end_idx = int(np.ceil(region.end_s * fs))
            start_idx = max(0, min(start_idx, labels.size))
            end_idx = max(0, min(end_idx, labels.size))
            labels[start_idx:end_idx] = region.label
        return labels

    def default_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        if self.csv_path is None:
            return Path("labeled.csv")
        return self.csv_path.with_name(f"{self.csv_path.stem}_labeled.csv")

    def save_labels(self) -> None:
        if self.data.size == 0:
            return

        suggested = self.default_output_path()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salva CSV etichettato",
            str(suggested),
            "CSV (*.csv);;Tutti i file (*)",
        )
        if not filename:
            return

        output = Path(filename)
        labels = self.labels_for_samples()
        try:
            write_labeled_csv(output, self.fieldnames, self.data, labels)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Errore", f"Non riesco a salvare:\n{exc}")
            return

        counts = {label: int(np.sum(labels == label)) for label in sorted(set(labels))}
        self.log(f"Salvato {output}")
        self.log(f"Conteggi label: {counts}")


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = CsvLabeler(
        csv_file=args.csv_file,
        fs=args.fs,
        initial_channel=args.channel,
        selected_label=args.label,
        background_label=args.background_label,
        output_path=args.output,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
