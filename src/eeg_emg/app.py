from __future__ import annotations

import sys
import time
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .acquisition import RateMonitor
from .config import BAUDRATE, CHANNEL_COUNT, FREQUENCIES
from .processing import (
    apply_butterworth_offline_zero_phase,
    apply_butterworth_realtime_causal,
    apply_butterworth_realtime_fixed_point,
    apply_fir_offline_zero_phase,
    apply_fir_realtime_fixed_point,
    design_butterworth,
    design_highpass,
    design_notch_50hz,
    fft_spectrum,
    fir_causal_fixed_point,
    moving_average,
    odd_taps,
    quantize_taps,
    quantize_sos,
    signal,
    sos_fixed_point_section_stats,
)
from .replay import CsvReplay
from .protocol import (
    ads2_enabled_command,
    frequency_command,
    gain_command,
    input_command,
    parse_channel_line,
    parse_register_line,
    read_registers_command,
    unipolar_command,
)
from .serial_worker import SerialWorker, list_ports
from .storage import write_csv
from .inference import LiveInferencePipeline
from .widgets.register_panel import RegisterPanel


class InferenceWorker(QtCore.QObject):
    result_ready = QtCore.Signal(object)
    model_loaded = QtCore.Signal(str)
    filter_updated = QtCore.Signal(bool)
    flush_ready = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, pipeline: LiveInferencePipeline) -> None:
        super().__init__()
        self.pipeline = pipeline

    @QtCore.Slot(object)
    def add_sample(self, sample: np.ndarray) -> None:
        try:
            result = self.pipeline.add_sample(sample)
            if result is not None:
                self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    @QtCore.Slot()
    def clear(self) -> None:
        self.pipeline.clear()

    @QtCore.Slot(str)
    def load_model(self, filename: str) -> None:
        try:
            self.pipeline.load_model(filename)
        except Exception as exc:
            self.error.emit(f"Caricamento modello fallito: {exc}")
            return
        self.filter_updated.emit(self.pipeline.filter_enabled)
        self.model_loaded.emit(filename)

    @QtCore.Slot(bool)
    def set_filter_enabled(self, enabled: bool) -> None:
        try:
            self.pipeline.set_filter_enabled(enabled)
        except Exception as exc:
            self.error.emit(f"Cambio filtro fallito: {exc}")
            return
        self.filter_updated.emit(self.pipeline.filter_enabled)

    @QtCore.Slot()
    def flush(self) -> None:
        """Acknowledge that all previously queued samples were processed."""
        self.flush_ready.emit()


class EEGEMGApp(QtWidgets.QMainWindow):
    connect_requested = QtCore.Signal(str)
    write_requested = QtCore.Signal(str)
    close_requested = QtCore.Signal()
    inference_sample_ready = QtCore.Signal(object)
    inference_clear_requested = QtCore.Signal()
    inference_model_requested = QtCore.Signal(str)
    inference_filter_requested = QtCore.Signal(bool)
    inference_flush_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEG EMG")
        self.resize(1530, 760)

        self.freq = 250
        self.sample_idx = 0
        self._data_revision = 0
        self._data_storage = np.empty((0, CHANNEL_COUNT), dtype=float)
        self._data_count = 0
        self.data = self._data_storage[:0]
        self._postproc_cache_signature: tuple[object, ...] | None = None
        self.offset = np.zeros(CHANNEL_COUNT, dtype=float)
        self.realtime_enabled = False
        self.is_recording = False
        self.live_channel = 0
        self._last_live_channel = 0
        self.connected = False
        self.recording_started_at = 0.0
        self.last_csv_path: Path | None = None
        self.settings = QtCore.QSettings("EEG_EMG", "EEG_EMG")
        self.log_lines = deque(maxlen=300)
        self.rate_monitor = RateMonitor(target_sps=self.freq)
        self.live_inference_enabled = False
        self.inference_results: list[dict[str, object]] = []
        self._pending_csv_save: tuple[Path, np.ndarray] | None = None
        self.inference_pipeline = LiveInferencePipeline(
            window_size=100,
            window_shift=25,
            n_channels=8,
            delta=500,
            max_derivative_order=0,
        )
        self.inference_thread = QtCore.QThread(self)
        self.inference_worker = InferenceWorker(self.inference_pipeline)
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_sample_ready.connect(self.inference_worker.add_sample)
        self.inference_clear_requested.connect(self.inference_worker.clear)
        self.inference_model_requested.connect(self.inference_worker.load_model)
        self.inference_filter_requested.connect(self.inference_worker.set_filter_enabled)
        self.inference_flush_requested.connect(self.inference_worker.flush)
        self.inference_worker.result_ready.connect(self.update_inference_output)
        self.inference_worker.model_loaded.connect(self.on_inference_model_loaded)
        self.inference_worker.filter_updated.connect(self.on_inference_filter_updated)
        self.inference_worker.flush_ready.connect(self._finish_save_csv)
        self.inference_worker.error.connect(self.on_inference_error)
        self.inference_thread.finished.connect(self.inference_worker.deleteLater)
        self.inference_thread.start()
        self.replay = CsvReplay(self)
        self.replay.sample_ready.connect(self.append_sample)
        self.replay.status.connect(self.show_status)
        self.replay.finished.connect(self.on_replay_finished)

        self.worker = SerialWorker()
        self.connect_requested.connect(self.worker.open)
        self.write_requested.connect(self.worker.write_line)
        self.close_requested.connect(self.worker.close)
        self.worker.line_received.connect(self.handle_serial_line)
        self.worker.connected.connect(self.set_connected)
        self.worker.error.connect(self.show_status)

        self._build_ui()
        self._make_plot_timer()

    def _build_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        self.tabs = tabs

        self.general_tab = QtWidgets.QWidget()
        self.channels_tab = QtWidgets.QWidget()
        self.registers_tab = QtWidgets.QWidget()
        self.fourier_tab = QtWidgets.QWidget()
        self.postproc_tab = QtWidgets.QWidget()
        tabs.addTab(self.general_tab, "General")
        tabs.addTab(self.channels_tab, "Channels")
        tabs.addTab(self.registers_tab, "Registers")
        tabs.addTab(self.fourier_tab, "Fourier")
        tabs.addTab(self.postproc_tab, "Post Processing")

        self._build_general_tab()
        self.channel_plots = self._build_plot_grid(self.channels_tab)
        self.fourier_plots = self._build_plot_grid(self.fourier_tab)
        self._build_registers_tab()
        self._build_postprocessing_tab()
        tabs.currentChanged.connect(self._on_tab_changed)

        self.statusBar().showMessage("Pronto.")

    def _build_general_tab(self) -> None:
        layout = QtWidgets.QGridLayout(self.general_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)

        self.live_plot = pg.PlotWidget()
        self.live_plot.setMinimumHeight(460)
        self.live_plot.setTitle("Live CH 1")
        self.live_plot.setLabel("bottom", "Time [s]")
        self.live_plot.setLabel("left", "Amplitude [V]")
        self.live_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.live_plot, 0, 0, 1, 5)

        self._build_general_controls(self.general_tab)

    def _build_plot_grid(self, parent: QtWidgets.QWidget) -> list[pg.PlotWidget]:
        layout = QtWidgets.QGridLayout(parent)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(10)
        plots: list[pg.PlotWidget] = []
        for idx in range(CHANNEL_COUNT):
            plot = pg.PlotWidget()
            plot.setTitle("Title")
            plot.setLabel("bottom", "X")
            plot.setLabel("left", "Y")
            plot.showGrid(x=True, y=True, alpha=0.2)
            row = idx // 4
            col = idx % 4
            layout.addWidget(plot, row, col)
            plots.append(plot)
        return plots

    def _build_general_controls(self, parent: QtWidgets.QWidget) -> None:
        layout = parent.layout()
        assert isinstance(layout, QtWidgets.QGridLayout)

        recording = QtWidgets.QGroupBox("Recording")
        recording_layout = QtWidgets.QGridLayout(recording)
        start_button = QtWidgets.QPushButton("START")
        start_button.setProperty("accent", True)
        stop_button = QtWidgets.QPushButton("STOP")
        realtime_check = QtWidgets.QCheckBox("Real Time Plot Enable")
        self.live_channel_combo = QtWidgets.QComboBox()
        self.live_channel_combo.addItems([f"CH{i}" for i in range(1, CHANNEL_COUNT + 1)])
        self.live_channel_combo.currentIndexChanged.connect(self.set_live_channel)
        start_button.clicked.connect(self.start_recording)
        stop_button.clicked.connect(self.stop_recording)
        realtime_check.toggled.connect(self.set_realtime)
        recording_layout.addWidget(start_button, 0, 0)
        recording_layout.addWidget(stop_button, 1, 0)
        recording_layout.addWidget(realtime_check, 0, 1)
        recording_layout.addWidget(QtWidgets.QLabel("Live CH"), 1, 1)
        recording_layout.addWidget(self.live_channel_combo, 1, 2)
        layout.addWidget(recording, 1, 0, 1, 2)

        save = QtWidgets.QGroupBox("DATA SAVE")
        save_layout = QtWidgets.QVBoxLayout(save)
        save_button = QtWidgets.QPushButton("SAVE")
        save_button.setProperty("accent", True)
        filter_button = QtWidgets.QPushButton("FILTER")
        save_button.clicked.connect(self.save_csv)
        filter_button.clicked.connect(self.filter_data)
        save_layout.addWidget(save_button)
        save_layout.addWidget(filter_button)
        layout.addWidget(save, 1, 2)

        serial_box = QtWidgets.QGroupBox("Serial communication")
        serial_layout = QtWidgets.QGridLayout(serial_box)
        self.com_list = QtWidgets.QListWidget()
        self.connection_lamp = QtWidgets.QLabel()
        self.connection_lamp.setFixedSize(34, 34)
        self.connection_lamp.setAutoFillBackground(True)
        self._set_lamp(False)
        search_button = QtWidgets.QPushButton("SEARCH")
        connect_button = QtWidgets.QPushButton("Connect")
        connect_button.setProperty("accent", True)
        reset_button = QtWidgets.QPushButton("Reset")
        search_button.clicked.connect(self.search_ports)
        connect_button.clicked.connect(self.connect_serial)
        reset_button.clicked.connect(lambda: self.show_status("Reset non implementato nel MATLAB originale."))
        serial_layout.addWidget(QtWidgets.QLabel("COM"), 0, 0)
        serial_layout.addWidget(self.com_list, 0, 1, 3, 1)
        serial_layout.addWidget(self.connection_lamp, 0, 2, alignment=QtCore.Qt.AlignCenter)
        serial_layout.addWidget(connect_button, 1, 3)
        serial_layout.addWidget(reset_button, 2, 3)
        serial_layout.addWidget(search_button, 3, 1)
        layout.addWidget(serial_box, 1, 3)

        self.inference_group = QtWidgets.QGroupBox("Live inference")
        inference_layout = QtWidgets.QGridLayout(self.inference_group)
        self.inference_enable_check = QtWidgets.QCheckBox("Enable")
        self.inference_enable_check.toggled.connect(self.set_live_inference_enabled)
        self.inference_filter_check = QtWidgets.QCheckBox("Filter")
        self.inference_filter_check.setChecked(self.inference_pipeline.filter_enabled)
        self.inference_filter_check.setToolTip(
            "Applica notch 50 Hz e passa-alto 5 Hz prima dell'encoding"
        )
        self.inference_filter_check.toggled.connect(self.set_inference_filter_enabled)
        self.inference_model_button = QtWidgets.QPushButton("Load model")
        self.inference_model_button.clicked.connect(self.load_inference_model)
        self.inference_model_label = QtWidgets.QLabel("Model: none")
        self.inference_model_label.setWordWrap(True)
        self.inference_label = QtWidgets.QLabel("Prediction: --")
        self.inference_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.inference_timing_label = QtWidgets.QLabel("Timing: --")
        self.inference_timing_label.setWordWrap(True)
        self.inference_spike_label = QtWidgets.QLabel("Spike rate: --")
        self.inference_spike_label.setWordWrap(True)
        inference_layout.addWidget(self.inference_enable_check, 0, 0)
        inference_layout.addWidget(self.inference_filter_check, 0, 1)
        inference_layout.addWidget(self.inference_model_button, 0, 2)
        inference_layout.addWidget(self.inference_model_label, 0, 3)
        inference_layout.addWidget(self.inference_label, 1, 0, 1, 4)
        inference_layout.addWidget(self.inference_timing_label, 2, 0, 1, 4)
        inference_layout.addWidget(self.inference_spike_label, 3, 0, 1, 4)
        layout.addWidget(self.inference_group, 1, 4)

        self.replay_group = QtWidgets.QGroupBox("CSV replay")
        replay_layout = QtWidgets.QGridLayout(self.replay_group)
        self.replay_load_button = QtWidgets.QPushButton("Load CSV")
        self.replay_load_button.clicked.connect(self.load_replay_csv)
        self.replay_start_button = QtWidgets.QPushButton("Start replay")
        self.replay_start_button.clicked.connect(self.start_replay)
        self.replay_stop_button = QtWidgets.QPushButton("Stop replay")
        self.replay_stop_button.clicked.connect(self.stop_replay)
        self.replay_stop_button.setEnabled(False)
        self.replay_fs_spin = QtWidgets.QDoubleSpinBox()
        self.replay_fs_spin.setRange(1.0, 16000.0)
        self.replay_fs_spin.setDecimals(1)
        self.replay_fs_spin.setValue(1000.0)
        self.replay_fs_spin.setSuffix(" Hz")
        self.replay_file_label = QtWidgets.QLabel("Replay: none")
        self.replay_file_label.setWordWrap(True)
        replay_layout.addWidget(self.replay_load_button, 0, 0)
        replay_layout.addWidget(self.replay_start_button, 0, 1)
        replay_layout.addWidget(self.replay_stop_button, 0, 2)
        replay_layout.addWidget(QtWidgets.QLabel("Fs"), 0, 3)
        replay_layout.addWidget(self.replay_fs_spin, 0, 4)
        replay_layout.addWidget(self.replay_file_label, 0, 5)
        replay_layout.setColumnStretch(5, 1)
        layout.addWidget(self.replay_group, 2, 0, 1, 5)

        self.search_ports()

    def load_inference_model(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Carica modello inferenza",
            str(Path.cwd()),
            "PyTorch model (*.pt *.pth);;Tutti i file (*)",
        )
        if not filename:
            return

        if self.live_inference_enabled:
            self.inference_enable_check.setChecked(False)
        self.inference_model_button.setEnabled(False)
        self.inference_model_label.setText(f"Model: caricamento {Path(filename).name}...")
        self.inference_model_requested.emit(filename)

    @QtCore.Slot(str)
    def on_inference_model_loaded(self, filename: str) -> None:
        self.inference_model_button.setEnabled(True)
        self.inference_model_label.setText(f"Model: {Path(filename).name}")
        self.show_status(f"Modello inferenza caricato: {filename}")

    @QtCore.Slot(str)
    def on_inference_error(self, message: str) -> None:
        self.inference_model_button.setEnabled(True)
        if self.live_inference_enabled:
            self.inference_enable_check.setChecked(False)
        self.show_status(f"Live inference disabilitata per errore: {message}")

    def set_live_inference_enabled(self, enabled: bool) -> None:
        self.live_inference_enabled = enabled
        if enabled:
            self.inference_clear_requested.emit()
            self.show_status("Live inference abilitata.")
        else:
            self.show_status("Live inference disabilitata.")

    def set_inference_filter_enabled(self, enabled: bool) -> None:
        self.inference_filter_requested.emit(enabled)

    @QtCore.Slot(bool)
    def on_inference_filter_updated(self, enabled: bool) -> None:
        self.inference_filter_check.blockSignals(True)
        self.inference_filter_check.setChecked(enabled)
        self.inference_filter_check.blockSignals(False)
        state = "abilitato" if enabled else "disabilitato"
        self.show_status(f"Filtro inferenza {state}; finestra reinizializzata.")

    def update_inference_output(self, result) -> None:
        self.inference_label.setText(f"Prediction: {result.class_name}")
        self.inference_timing_label.setText(
            f"Timing: enc={result.encoding_ms:.2f} ms | "
            f"infer={result.inference_ms:.2f} ms | total={result.total_ms:.2f} ms"
        )
        self.inference_spike_label.setText(f"Spike rate: {result.spike_rate:.5f}")
        self.store_inference_result(result)

    def store_inference_result(self, result) -> None:
        scores = result.scores.tolist() if result.scores is not None else []
        true_label = ""
        if hasattr(self, "replay") and self.replay.labels is not None and self.sample_idx > 0:
            label_idx = min(self.sample_idx - 1, len(self.replay.labels) - 1)
            true_label = str(self.replay.labels[label_idx])

        self.inference_results.append({
            "app_sample_index": int(self.sample_idx),
            "pipeline_sample_index": int(result.sample_index),
            "true_label": true_label,
            "predicted_class": "" if result.predicted_class is None else int(result.predicted_class),
            "predicted_label": result.class_name,
            "score_0": scores[0] if len(scores) > 0 else "",
            "score_1": scores[1] if len(scores) > 1 else "",
            "score_2": scores[2] if len(scores) > 2 else "",
            "score_3": scores[3] if len(scores) > 3 else "",
            "encoding_ms": float(result.encoding_ms),
            "inference_ms": float(result.inference_ms),
            "total_ms": float(result.total_ms),
            "spike_rate": float(result.spike_rate),
        })

    def load_replay_csv(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Carica CSV replay",
            str(Path.cwd()),
            "CSV (*.csv);;Tutti i file (*)",
        )
        if not filename:
            return

        try:
            self.replay.load_csv(filename)
        except Exception as exc:
            self.show_status(f"Caricamento replay fallito: {exc}")
            return

        self.replay_file_label.setText(f"Replay: {Path(filename).name}")

    def start_replay(self) -> None:
        try:
            self.sample_idx = 0
            self._reset_sample_storage()
            self.inference_results = []
            self.clear_time_plots()
            if self.realtime_enabled:
                self.prepare_live_plot()
            self.inference_clear_requested.emit()
            self.replay.start(self.replay_fs_spin.value())
        except Exception as exc:
            self.show_status(f"Replay non avviato: {exc}")
            return

        self.replay_start_button.setEnabled(False)
        self.replay_stop_button.setEnabled(True)

    def stop_replay(self) -> None:
        self.replay.stop()

    def on_replay_finished(self) -> None:
        if hasattr(self, "replay_start_button"):
            self.replay_start_button.setEnabled(True)
            self.replay_stop_button.setEnabled(False)
        self.plot_time_domain()
        self.plot_frequency_domain()
        self.update_postprocessing_plot()
        self.prompt_save_inference_results()

    def prompt_save_inference_results(self) -> None:
        if not self.inference_results:
            self.show_status("Replay finito: nessuna predizione da salvare.")
            return

        default_name = f"replay_predictions_{datetime.now().strftime('%d_%b_%Y_%H_%M_%S')}.csv"
        path = self._prompt_save_path(default_name, "Salva predizioni replay")
        if path is None:
            self.show_status("Replay finito: salvataggio predizioni annullato.")
            return

        fieldnames = list(self.inference_results[0].keys())
        try:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.inference_results)
        except Exception as exc:
            self.show_status(f"Salvataggio predizioni fallito: {exc}")
            return

        self.show_status(f"Predizioni replay salvate: {path}")

    def _build_registers_tab(self) -> None:
        layout = QtWidgets.QGridLayout(self.registers_tab)
        self.ads1_panel = RegisterPanel("ADS1")
        self.ads2_panel = RegisterPanel("ADS2")
        for panel in (self.ads1_panel, self.ads2_panel):
            panel.read_all_requested.connect(self.read_all_registers)
            panel.update_channel_requested.connect(self.update_channel)
            panel.update_all_requested.connect(self.update_all_channels)
            panel.unipolar_changed.connect(self.set_unipolar)
        layout.addWidget(self.ads1_panel, 0, 0)
        layout.addWidget(self.ads2_panel, 0, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.frequency_combo = QtWidgets.QComboBox()
        self.frequency_combo.addItems(FREQUENCIES.keys())
        self.frequency_combo.setCurrentText("250SPS")
        self.frequency_combo.currentTextChanged.connect(self.set_frequency)
        noise_button = QtWidgets.QPushButton("Noise")
        offset_button = QtWidgets.QPushButton("OFFSET")
        self.ads2_on_check = QtWidgets.QCheckBox("ADS2 ON/OFF")
        noise_button.clicked.connect(self.compute_noise)
        offset_button.clicked.connect(self.compute_offset)
        self.ads2_on_check.toggled.connect(self.set_ads2_enabled)

        bottom.addWidget(QtWidgets.QLabel("Frequency"))
        bottom.addWidget(self.frequency_combo)
        bottom.addWidget(noise_button)
        bottom.addWidget(offset_button)
        bottom.addStretch(1)
        bottom.addWidget(self.ads2_on_check)
        layout.addLayout(bottom, 1, 0, 1, 2)

        self.serial_log = QtWidgets.QPlainTextEdit()
        self.serial_log.setReadOnly(True)
        self.serial_log.setMaximumBlockCount(300)
        layout.addWidget(self.serial_log, 2, 0, 1, 2)

    def _build_postprocessing_tab(self) -> None:
        layout = QtWidgets.QGridLayout(self.postproc_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)

        self.postproc_plot = pg.PlotWidget()
        self.postproc_plot.setMinimumHeight(360)
        self.postproc_plot.setLabel("bottom", "Time [s]")
        self.postproc_plot.setLabel("left", "Amplitude [counts]")
        self.postproc_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.postproc_plot, 0, 0, 1, 6)

        self.postproc_response_plot = pg.PlotWidget()
        self.postproc_response_plot.setMinimumHeight(220)
        self.postproc_response_plot.setLabel("bottom", "Frequency [Hz]")
        self.postproc_response_plot.setLabel("left", "Magnitude [dB]")
        self.postproc_response_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.postproc_response_plot, 1, 0, 1, 3)

        self.postproc_zplane_plot = pg.PlotWidget()
        self.postproc_zplane_plot.setMinimumHeight(220)
        self.postproc_zplane_plot.setAspectLocked(True)
        self.postproc_zplane_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.postproc_zplane_plot, 1, 3, 1, 2)

        self.postproc_console = QtWidgets.QPlainTextEdit()
        self.postproc_console.setReadOnly(True)
        self.postproc_console.setMaximumBlockCount(120)
        self.postproc_console.setMinimumHeight(220)
        layout.addWidget(self.postproc_console, 1, 5)

        controls = QtWidgets.QGroupBox("Elaborazione (non distruttiva sui dati registrati)")
        controls_layout = QtWidgets.QGridLayout(controls)

        load_button = QtWidgets.QPushButton("Carica CSV...")
        load_button.clicked.connect(self.load_postprocessing_csv)
        controls_layout.addWidget(load_button, 0, 0)

        self.postproc_channel_combo = QtWidgets.QComboBox()
        self.postproc_channel_combo.addItems([f"CH{i}" for i in range(1, CHANNEL_COUNT + 1)])
        self.postproc_channel_combo.currentIndexChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(QtWidgets.QLabel("Canale"), 0, 1)
        controls_layout.addWidget(self.postproc_channel_combo, 0, 2)

        self.postproc_fs_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_fs_spin.setRange(1.0, 20000.0)
        self.postproc_fs_spin.setValue(float(self.freq))
        self.postproc_fs_spin.setDecimals(2)
        self.postproc_fs_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(QtWidgets.QLabel("Fs [Hz]"), 0, 3)
        controls_layout.addWidget(self.postproc_fs_spin, 0, 4)

        self.postproc_show_raw_check = QtWidgets.QCheckBox("Mostra grezzo")
        self.postproc_show_raw_check.setChecked(True)
        self.postproc_show_raw_check.toggled.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_show_raw_check, 0, 5)

        self.postproc_offset_check = QtWidgets.QCheckBox("Rimuovi offset (media)")
        self.postproc_offset_check.toggled.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_offset_check, 1, 0, 1, 2)
        self.postproc_offset_check.hide()

        self.postproc_movavg_check = QtWidgets.QCheckBox("Media mobile - finestra [campioni]")
        self.postproc_movavg_check.toggled.connect(self.update_postprocessing_plot)
        self.postproc_movavg_spin = QtWidgets.QSpinBox()
        self.postproc_movavg_spin.setRange(2, 5000)
        self.postproc_movavg_spin.setValue(5)
        self.postproc_movavg_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_movavg_check, 1, 2, 1, 2)
        controls_layout.addWidget(self.postproc_movavg_spin, 1, 4)
        self.postproc_movavg_check.hide()
        self.postproc_movavg_spin.hide()

        self.postproc_notch_check = QtWidgets.QCheckBox("Notch")
        self.postproc_notch_check.setChecked(True)
        self.postproc_notch_check.toggled.connect(self.update_postprocessing_plot)
        self.postproc_notch_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_notch_spin.setRange(1.0, 1000.0)
        self.postproc_notch_spin.setValue(50.0)
        self.postproc_notch_spin.setDecimals(2)
        self.postproc_notch_spin.valueChanged.connect(self.update_postprocessing_plot)
        self.postproc_notch_bw_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_notch_bw_spin.setRange(0.1, 50.0)
        self.postproc_notch_bw_spin.setValue(2.0)
        self.postproc_notch_bw_spin.setDecimals(2)
        self.postproc_notch_bw_spin.valueChanged.connect(self.update_postprocessing_plot)
        self.postproc_notch_q_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_notch_q_spin.setRange(1.0, 500.0)
        self.postproc_notch_q_spin.setValue(30.0)
        self.postproc_notch_q_spin.setDecimals(1)
        self.postproc_notch_q_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_notch_check, 2, 0)
        controls_layout.addWidget(QtWidgets.QLabel("f [Hz]"), 2, 1)
        controls_layout.addWidget(self.postproc_notch_spin, 2, 2)
        controls_layout.addWidget(QtWidgets.QLabel("FIR bw"), 2, 3)
        controls_layout.addWidget(self.postproc_notch_bw_spin, 2, 4)
        controls_layout.addWidget(QtWidgets.QLabel("IIR Q"), 2, 5)
        controls_layout.addWidget(self.postproc_notch_q_spin, 2, 6)
        for col in range(7):
            item = controls_layout.itemAtPosition(2, col)
            if item and item.widget():
                item.widget().hide()

        self.postproc_highpass_check = QtWidgets.QCheckBox("Passa-alto - taglio [Hz]")
        self.postproc_highpass_check.setChecked(True)
        self.postproc_highpass_check.toggled.connect(self.update_postprocessing_plot)
        self.postproc_highpass_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_highpass_spin.setRange(0.05, 500.0)
        self.postproc_highpass_spin.setValue(20.0)
        self.postproc_highpass_spin.setDecimals(2)
        self.postproc_highpass_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_highpass_check, 3, 0, 1, 2)
        controls_layout.addWidget(self.postproc_highpass_spin, 3, 2)

        self.postproc_lowpass_check = QtWidgets.QCheckBox("Passa-basso - taglio [Hz]")
        self.postproc_lowpass_check.toggled.connect(self.update_postprocessing_plot)
        self.postproc_lowpass_spin = QtWidgets.QDoubleSpinBox()
        self.postproc_lowpass_spin.setRange(1.0, 8000.0)
        self.postproc_lowpass_spin.setValue(450.0)
        self.postproc_lowpass_spin.setDecimals(2)
        self.postproc_lowpass_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(self.postproc_lowpass_check, 3, 3, 1, 2)
        controls_layout.addWidget(self.postproc_lowpass_spin, 3, 5)
        for col in range(6):
            item = controls_layout.itemAtPosition(3, col)
            if item and item.widget():
                item.widget().hide()

        self.postproc_fir_taps_spin = QtWidgets.QSpinBox()
        self.postproc_fir_taps_spin.setRange(5, 5001)
        self.postproc_fir_taps_spin.setSingleStep(2)
        self.postproc_fir_taps_spin.setValue(101)
        self.postproc_fir_taps_spin.valueChanged.connect(self.update_postprocessing_plot)
        self.postproc_butter_order_spin = QtWidgets.QSpinBox()
        self.postproc_butter_order_spin.setRange(1, 12)
        self.postproc_butter_order_spin.setValue(4)
        self.postproc_butter_order_spin.valueChanged.connect(self.update_postprocessing_plot)
        self.postproc_frac_bits_spin = QtWidgets.QSpinBox()
        self.postproc_frac_bits_spin.setRange(4, 30)
        self.postproc_frac_bits_spin.setValue(15)
        self.postproc_frac_bits_spin.valueChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(QtWidgets.QLabel("FIR taps"), 4, 0)
        controls_layout.addWidget(self.postproc_fir_taps_spin, 4, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Butter order"), 4, 2)
        controls_layout.addWidget(self.postproc_butter_order_spin, 4, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Frac bits"), 4, 4)
        controls_layout.addWidget(self.postproc_frac_bits_spin, 4, 5)
        for col in range(6):
            item = controls_layout.itemAtPosition(4, col)
            if item and item.widget():
                item.widget().hide()

        self.postproc_filter_checks: dict[str, QtWidgets.QCheckBox] = {}
        filter_labels = [
            ("fir_offline", "FIR offline"),
            ("fir_realtime", "FIR realtime fixed"),
            ("butter_offline", "Butter offline"),
            ("butter_realtime", "Butter realtime"),
            ("butter_fixed", "Butter fixed"),
        ]
        for col, (key, label) in enumerate(filter_labels):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(key in {"butter_realtime", "butter_fixed"})
            check.toggled.connect(self.update_postprocessing_plot)
            self.postproc_filter_checks[key] = check
            controls_layout.addWidget(check, 5, col)
            check.hide()

        self.postproc_diagnostic_combo = QtWidgets.QComboBox()
        self.postproc_diagnostic_combo.addItems([label for _, label in filter_labels])
        self.postproc_diagnostic_combo.setCurrentText("Butter fixed")
        self.postproc_diagnostic_combo.currentIndexChanged.connect(self.update_postprocessing_plot)
        controls_layout.addWidget(QtWidgets.QLabel("Zeri/poli per"), 5, 5)
        controls_layout.addWidget(self.postproc_diagnostic_combo, 5, 6)
        controls_layout.itemAtPosition(5, 5).widget().hide()
        self.postproc_diagnostic_combo.hide()

        self.postproc_filter_combo = QtWidgets.QComboBox()
        self.postproc_filter_items = [
            ("none", "Nessun filtro"),
            ("fir_offline", "FIR offline"),
            ("fir_realtime", "FIR realtime fixed"),
            ("butter_offline", "Butterworth offline"),
            ("butter_realtime", "Butterworth realtime"),
            ("butter_fixed", "Butterworth fixed"),
        ]
        for _, label in self.postproc_filter_items:
            self.postproc_filter_combo.addItem(label)
        self.postproc_filter_combo.currentIndexChanged.connect(self._on_postproc_filter_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Filtro"), 1, 0)
        controls_layout.addWidget(self.postproc_filter_combo, 1, 1, 1, 2)

        self.postproc_raw_always_label = QtWidgets.QLabel("Disattiva 'Mostra grezzo' per vedere solo il filtro.")
        controls_layout.addWidget(self.postproc_raw_always_label, 1, 3, 1, 4)

        self.postproc_param_stack = QtWidgets.QStackedWidget()
        self.postproc_param_pages: dict[str, QtWidgets.QWidget] = {}
        self._build_postproc_param_pages()
        controls_layout.addWidget(self.postproc_param_stack, 2, 0, 4, 7)
        self._on_postproc_filter_changed()

        if signal is None:
            for widget in (
                self.postproc_notch_check,
                self.postproc_highpass_check,
                self.postproc_highpass_spin,
                self.postproc_lowpass_check,
                self.postproc_lowpass_spin,
                self.postproc_notch_spin,
                self.postproc_notch_bw_spin,
                self.postproc_notch_q_spin,
                self.postproc_fir_taps_spin,
                self.postproc_butter_order_spin,
                self.postproc_frac_bits_spin,
            ):
                widget.setEnabled(False)
            controls_layout.addWidget(
                QtWidgets.QLabel("scipy non installato: filtri disabilitati"), 6, 0, 1, 3
            )

        layout.addWidget(controls, 2, 0, 1, 6)

        save_box = QtWidgets.QGroupBox("Salvataggio")
        save_layout = QtWidgets.QHBoxLayout(save_box)
        save_raw_button = QtWidgets.QPushButton("Salva dati grezzi...")
        save_processed_button = QtWidgets.QPushButton("Salva dati elaborati...")
        save_raw_button.clicked.connect(self.save_postprocessing_raw)
        save_processed_button.clicked.connect(self.save_postprocessing_processed)
        save_layout.addWidget(save_raw_button)
        save_layout.addWidget(save_processed_button)
        layout.addWidget(save_box, 3, 0, 1, 6)

    def _build_postproc_param_pages(self) -> None:
        none_page = QtWidgets.QWidget()
        self.postproc_param_layout = QtWidgets.QGridLayout(none_page)
        self.postproc_param_layout.setContentsMargins(0, 0, 0, 0)
        self.postproc_param_layout.setHorizontalSpacing(12)
        self.postproc_param_layout.setVerticalSpacing(8)
        self.postproc_param_pages["dynamic"] = none_page
        self.postproc_param_stack.addWidget(none_page)

    def _clear_postproc_param_layout(self) -> None:
        shared = {
            self.postproc_notch_spin,
            self.postproc_notch_bw_spin,
            self.postproc_notch_q_spin,
            self.postproc_highpass_spin,
            self.postproc_lowpass_spin,
            self.postproc_fir_taps_spin,
            self.postproc_butter_order_spin,
            self.postproc_frac_bits_spin,
            self.postproc_movavg_spin,
        }
        while self.postproc_param_layout.count():
            item = self.postproc_param_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in shared:
                widget.setParent(None)
            else:
                widget.deleteLater()

    def _add_proxy_check(
        self,
        row: int,
        col: int,
        label: str,
        target: QtWidgets.QCheckBox,
    ) -> QtWidgets.QCheckBox:
        check = QtWidgets.QCheckBox(label)
        check.setChecked(target.isChecked())
        check.toggled.connect(target.setChecked)
        check.toggled.connect(self.update_postprocessing_plot)
        self.postproc_param_layout.addWidget(check, row, col)
        return check

    def _populate_postproc_param_page(self, key: str) -> None:
        self._clear_postproc_param_layout()
        layout = self.postproc_param_layout
        if key == "none":
            layout.addWidget(QtWidgets.QLabel("Seleziona un filtro dal menu per mostrare i parametri."), 0, 0, 1, 4)
            layout.setColumnStretch(6, 1)
            return

        self._add_proxy_check(0, 0, "Notch", self.postproc_notch_check)
        layout.addWidget(QtWidgets.QLabel("f [Hz]"), 0, 1)
        layout.addWidget(self.postproc_notch_spin, 0, 2)
        self._add_proxy_check(1, 0, "Passa-alto", self.postproc_highpass_check)
        layout.addWidget(QtWidgets.QLabel("HP [Hz]"), 1, 1)
        layout.addWidget(self.postproc_highpass_spin, 1, 2)
        self._add_proxy_check(2, 0, "Passa-basso", self.postproc_lowpass_check)
        layout.addWidget(QtWidgets.QLabel("LP [Hz]"), 2, 1)
        layout.addWidget(self.postproc_lowpass_spin, 2, 2)
        self._add_proxy_check(3, 0, "Rimuovi offset", self.postproc_offset_check)
        self._add_proxy_check(3, 1, "Media mobile", self.postproc_movavg_check)
        layout.addWidget(self.postproc_movavg_spin, 3, 2)

        if key.startswith("fir"):
            layout.addWidget(QtWidgets.QLabel("FIR taps"), 0, 3)
            layout.addWidget(self.postproc_fir_taps_spin, 0, 4)
            layout.addWidget(QtWidgets.QLabel("Notch bw [Hz]"), 1, 3)
            layout.addWidget(self.postproc_notch_bw_spin, 1, 4)
            if key == "fir_realtime":
                layout.addWidget(QtWidgets.QLabel("Frac bits"), 2, 3)
                layout.addWidget(self.postproc_frac_bits_spin, 2, 4)
        else:
            layout.addWidget(QtWidgets.QLabel("Butter order"), 0, 3)
            layout.addWidget(self.postproc_butter_order_spin, 0, 4)
            layout.addWidget(QtWidgets.QLabel("Notch Q"), 1, 3)
            layout.addWidget(self.postproc_notch_q_spin, 1, 4)
            if key == "butter_fixed":
                layout.addWidget(QtWidgets.QLabel("Frac bits"), 2, 3)
                layout.addWidget(self.postproc_frac_bits_spin, 2, 4)
        layout.setColumnStretch(6, 1)

    def _selected_postproc_filter_key(self) -> str:
        if not hasattr(self, "postproc_filter_combo"):
            return "none"
        index = self.postproc_filter_combo.currentIndex()
        if index < 0 or index >= len(self.postproc_filter_items):
            return "none"
        return self.postproc_filter_items[index][0]

    def _on_postproc_filter_changed(self) -> None:
        key = self._selected_postproc_filter_key()
        self._populate_postproc_param_page(key)
        page = self.postproc_param_pages.get("dynamic")
        if page is not None:
            self.postproc_param_stack.setCurrentWidget(page)
        if hasattr(self, "postproc_diagnostic_combo"):
            label_by_key = {
                "fir_offline": "FIR offline",
                "fir_realtime": "FIR realtime fixed",
                "butter_offline": "Butter offline",
                "butter_realtime": "Butter realtime",
                "butter_fixed": "Butter fixed",
            }
            label = label_by_key.get(key)
            if label:
                self.postproc_diagnostic_combo.setCurrentText(label)
        self.update_postprocessing_plot()

    def _make_plot_timer(self) -> None:
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setInterval(250)
        self.plot_timer.timeout.connect(self.update_realtime_plots)
        self.plot_timer.start()

    def search_ports(self) -> None:
        self.com_list.clear()
        if list_ports is None:
            self.show_status("pyserial non e installato: impossibile cercare le COM.")
            return
        ports = [port.device for port in list_ports.comports()]
        self.com_list.addItems(ports)
        if ports:
            self.com_list.setCurrentRow(0)

    def connect_serial(self) -> None:
        item = self.com_list.currentItem()
        if item is None:
            self.show_status("Seleziona una porta COM prima di connettere.")
            return
        self.connect_requested.emit(item.text())

    def send_command(self, command: str) -> None:
        self.log_serial(f"> {command}")
        self.write_requested.emit(command)

    def start_recording(self) -> None:
        self.sample_idx = 0
        self._reset_sample_storage()
        self.inference_results = []
        if self.live_inference_enabled:
            self.inference_clear_requested.emit()
        self.is_recording = True
        self.recording_started_at = time.monotonic()
        print(self.rate_monitor.reset(self.freq), flush=True)
        self.clear_time_plots()
        if self.realtime_enabled:
            self.prepare_live_plot()
        self.send_command("RDATAC")

    def stop_recording(self) -> None:
        self.send_command("STOP")
        self.is_recording = False
        if self.recording_started_at:
            elapsed = time.monotonic() - self.recording_started_at
            self.show_status(f"Registrazione fermata. Durata: {elapsed:.2f} s.")
        summary = self.rate_monitor.summary()
        if summary:
            print(summary, flush=True)
        self.plot_time_domain()
        self.plot_frequency_domain()
        self.update_postprocessing_plot()

    def set_realtime(self, enabled: bool) -> None:
        self.realtime_enabled = enabled
        if enabled:
            self.prepare_live_plot()

    def set_live_channel(self, index: int) -> None:
        self.live_channel = index
        self.prepare_live_plot()

    def prepare_live_plot(self) -> None:
        if not hasattr(self, "live_plot"):
            return
        self._last_live_channel = self.live_channel
        plot = self.live_plot
        plot.clear()
        plot.setTitle(f"Live CH {self.live_channel + 1}")
        plot.setLabel("bottom", "Time [s]")
        plot.setLabel("left", "Amplitude [V]")

    def read_all_registers(self, ads_name: str) -> None:
        self.send_command(read_registers_command(ads_name))

    def update_channel(self, ads_name: str) -> None:
        panel = self.panel_for_ads(ads_name)
        ch = panel.selected_channel
        gain = panel.pga[ch - 1]
        self.send_command(gain_command(ads_name, ch, gain))
        self.send_command(input_command(ads_name, ch, panel.input_mode))

    def update_all_channels(self, ads_name: str) -> None:
        panel = self.panel_for_ads(ads_name)
        for ch in range(1, 9):
            gain = int(panel.pga_combo.currentText()) if panel.pga_combo.currentText().isdigit() else panel.pga[ch - 1]
            panel.pga[ch - 1] = gain
            self.send_command(gain_command(ads_name, ch, gain))
            self.send_command(input_command(ads_name, ch, panel.input_mode))
        self.read_all_registers(ads_name)

    def set_unipolar(self, ads_name: str, enabled: bool) -> None:
        self.send_command(unipolar_command(ads_name, enabled))

    def set_ads2_enabled(self, enabled: bool) -> None:
        self.send_command(ads2_enabled_command(enabled))

    def set_frequency(self, label: str) -> None:
        self.freq = FREQUENCIES[label]
        self.send_command(frequency_command(self.freq))

    def compute_noise(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato acquisito: impossibile calcolare il rumore.")
            return
        noise = (np.nanmax(self.data, axis=0) - np.nanmin(self.data, axis=0)) / 6.6
        self.ads1_panel.set_noise(noise[:8])
        if self.data.shape[1] >= 16:
            self.ads2_panel.set_noise(noise[8:16])

    def compute_offset(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato acquisito: impossibile calcolare l'offset.")
            return
        self.offset[: self.data.shape[1]] = np.nanmean(self.data, axis=0)
        self.show_status("Offset aggiornato.")

    def save_csv(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato da salvare.")
            return
        if self._pending_csv_save is not None:
            self.show_status("Salvataggio gia in corso: attendi il completamento dell'inferenza.")
            return
        stamp = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        path = self._prompt_save_path(f"{self.freq}_{stamp}.csv", "Salva acquisizione")
        if path is None:
            self.show_status("Salvataggio annullato.")
            return
        corrected = self.data.copy()
        corrected[:, : corrected.shape[1]] -= self.offset[: corrected.shape[1]]
        if corrected.shape[1] < CHANNEL_COUNT:
            corrected = np.pad(corrected, ((0, 0), (0, CHANNEL_COUNT - corrected.shape[1])))
        self._pending_csv_save = (path, corrected[:, :CHANNEL_COUNT])
        self.show_status("Attendo le ultime predizioni prima di salvare...")
        self.inference_flush_requested.emit()

    @QtCore.Slot()
    def _finish_save_csv(self) -> None:
        if self._pending_csv_save is None:
            return
        path, data = self._pending_csv_save
        self._pending_csv_save = None

        prediction_fields = [
            "prediction_source_sample",
            "prediction_is_new",
            "predicted_class",
            "predicted_label",
            "score_0",
            "score_1",
            "score_2",
            "score_3",
            "spike_rate",
            "inference_ms",
        ]
        fieldnames = ["sample_index", "time_s"] + [
            f"CH{idx}" for idx in range(1, CHANNEL_COUNT + 1)
        ] + prediction_fields
        predictions = sorted(
            self.inference_results,
            key=lambda item: int(item["pipeline_sample_index"]),
        )

        try:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                prediction_idx = 0
                current_prediction: dict[str, object] | None = None
                for sample_idx, sample in enumerate(data, start=1):
                    is_new = False
                    while (
                        prediction_idx < len(predictions)
                        and int(predictions[prediction_idx]["pipeline_sample_index"]) <= sample_idx
                    ):
                        current_prediction = predictions[prediction_idx]
                        is_new = (
                            int(current_prediction["pipeline_sample_index"]) == sample_idx
                        )
                        prediction_idx += 1

                    row: dict[str, object] = {
                        "sample_index": sample_idx,
                        "time_s": (sample_idx - 1) / self.freq,
                    }
                    row.update({f"CH{idx + 1}": value for idx, value in enumerate(sample)})
                    row.update({field: "" for field in prediction_fields})
                    if current_prediction is not None:
                        row.update({
                            "prediction_source_sample": current_prediction["pipeline_sample_index"],
                            "prediction_is_new": int(is_new),
                            "predicted_class": current_prediction["predicted_class"],
                            "predicted_label": current_prediction["predicted_label"],
                            "score_0": current_prediction["score_0"],
                            "score_1": current_prediction["score_1"],
                            "score_2": current_prediction["score_2"],
                            "score_3": current_prediction["score_3"],
                            "spike_rate": current_prediction["spike_rate"],
                            "inference_ms": current_prediction["inference_ms"],
                        })
                    writer.writerow(row)
        except Exception as exc:
            self.show_status(f"Salvataggio CSV fallito: {exc}")
            return

        self.last_csv_path = path
        self.show_status(
            f"Salvato: {path} ({len(data)} campioni, {len(predictions)} predizioni)"
        )


    def _prompt_save_path(
        self,
        default_name: str,
        title: str = "Salva CSV",
    ) -> Path | None:
        last_directory = self.settings.value("paths/last_save_directory", "", type=str)
        initial_path = str(Path(last_directory) / default_name) if last_directory else default_name
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title, initial_path, "CSV (*.csv)"
        )
        if not path_str:
            return None
        path = Path(path_str)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        self.settings.setValue("paths/last_save_directory", str(path.parent))
        return path

    def save_postprocessing_raw(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato da salvare.")
            return
        default_name = f"raw_{datetime.now().strftime('%d_%b_%Y_%H_%M_%S')}.csv"
        path = self._prompt_save_path(default_name)
        if path is None:
            return
        write_csv(path, self.data)
        self.show_status(f"Dati grezzi salvati: {path}")

    def save_postprocessing_processed(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato da salvare.")
            return
        processed = np.zeros_like(self.data)
        params = self._postproc_params()
        for idx in range(self.data.shape[1]):
            outputs, _ = self._selected_postproc_outputs(self.data[:, idx], params)
            processed[:, idx] = next(iter(outputs.values())) if outputs else self.data[:, idx]
        default_name = f"processed_{datetime.now().strftime('%d_%b_%Y_%H_%M_%S')}.csv"
        path = self._prompt_save_path(default_name)
        if path is None:
            return
        write_csv(path, processed)
        self.show_status(f"Dati elaborati salvati: {path}")

    def filter_data(self) -> None:
        if self.data.size == 0:
            self.show_status("Nessun dato da filtrare.")
            return
        if signal is None:
            self.show_status("scipy non e installato: filtro non disponibile.")
            return
        fs = self.freq
        try:
            notch = design_notch_50hz(fs, 201)
            hp_b, hp_a = design_butterworth(fs, 1, "highpass", order=4)
            limit = self.data.shape[1]
            for idx in range(limit):
                x = self.data[:, idx] - self.offset[idx]
                x = signal.filtfilt(notch, [1.0], x)
                x = signal.filtfilt(hp_b, hp_a, x)
                self.data[:, idx] = x
            self._mark_data_changed()
            self.plot_time_domain()
            self.plot_frequency_domain()
            self.show_status("Filtro applicato.")
        except Exception as exc:
            self.show_status(f"Filtro fallito: {exc}")

    @QtCore.Slot(str)
    def handle_serial_line(self, line: str) -> None:
        if not line:
            return
        if not line.startswith(("Ch1", "Ch2")):
            self.log_serial(line)
        if line.startswith("Status:"):
            self.sample_idx += 1
            return
        if line.startswith("Ch1"):
            values = self._parse_channel_line(line, expected=8, scale=False)
            if values is not None:
                self.append_sample(values)
            return
        if line.startswith("Ch2"):
            values = self._parse_channel_line(line, expected=16, scale=True)
            if values is not None:
                self.append_sample(values)
            return
        register = parse_register_line(line)
        if register:
            ads_name, address, value = register
            self.panel_for_ads(ads_name).set_register(address, value)

    def _parse_channel_line(self, line: str, expected: int, scale: bool) -> np.ndarray | None:
        try:
            return parse_channel_line(line, expected, scale)
        except ValueError as exc:
            self.show_status(str(exc) if str(exc) else f"Formato dati non valido: {line}")
            return None

    def append_sample(self, sample: np.ndarray) -> None:
        self.sample_idx += 1
        sample = np.asarray(sample, dtype=float)
        if sample.shape != (self._data_storage.shape[1],):
            raise ValueError(
                f"Campione con shape {sample.shape}, attesa {(self._data_storage.shape[1],)}"
            )
        if self._data_count >= len(self._data_storage):
            old_capacity = len(self._data_storage)
            new_capacity = max(4096, old_capacity * 2)
            storage = np.empty((new_capacity, self._data_storage.shape[1]), dtype=float)
            if self._data_count:
                storage[: self._data_count] = self._data_storage[: self._data_count]
            self._data_storage = storage
        self._data_storage[self._data_count] = sample
        self._data_count += 1
        self.data = self._data_storage[: self._data_count]
        self._mark_data_changed()

        if self.live_inference_enabled:
            self.inference_sample_ready.emit(sample[:8].copy())

        if self.is_recording:
            message = self.rate_monitor.record_sample()
            if message:
                print(message, flush=True)

    def clear_time_plots(self) -> None:
        if hasattr(self, "live_plot"):
            self.live_plot.clear()
            self.live_plot.setTitle(f"Live CH {self.live_channel + 1}")
            self.live_plot.setLabel("bottom", "Time [s]")
            self.live_plot.setLabel("left", "Amplitude [V]")
        for plot in self.channel_plots:
            plot.clear()
            plot.setTitle("Title")
            plot.setLabel("bottom", "X")
            plot.setLabel("left", "Y")

    def _reset_sample_storage(self, channel_count: int = CHANNEL_COUNT) -> None:
        self._data_storage = np.empty((4096, channel_count), dtype=float)
        self._data_count = 0
        self.data = self._data_storage[:0]
        self._mark_data_changed()

    def _set_sample_data(self, data: np.ndarray) -> None:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"Dati attesi in matrice 2D, ricevuta shape {values.shape}")
        capacity = max(4096, len(values))
        self._data_storage = np.empty((capacity, values.shape[1]), dtype=float)
        self._data_storage[: len(values)] = values
        self._data_count = len(values)
        self.data = self._data_storage[: self._data_count]
        self._mark_data_changed()

    def _mark_data_changed(self) -> None:
        self._data_revision += 1
        self._postproc_cache_signature = None

    def plot_time_domain(self) -> None:
        if self.data.size == 0:
            return
        t = np.arange(1, len(self.data) + 1) / self.freq
        for idx, plot in enumerate(self.channel_plots):
            plot.clear()
            if idx >= self.data.shape[1]:
                continue
            item = plot.plot(t, self.data[:, idx], pen=pg.mkPen((59, 130, 246), width=2))
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
            plot.setTitle(f"Ch {idx + 1}")
            plot.setLabel("bottom", "Time [s]")
            plot.setLabel("left", "Amplitude [V]")
            plot.setXRange(0, max(float(t[-1]), 1 / self.freq), padding=0)

    def update_realtime_plots(self) -> None:
        replay_running = hasattr(self, "replay") and self.replay.is_running()
        if not (self.is_recording or replay_running) or not self.realtime_enabled or self.data.size == 0:
            return
        window = min(len(self.data), max(self.freq * 2, 100))
        subset = self.data[-window:]
        t = np.arange(len(self.data) - window + 1, len(self.data) + 1) / self.freq
        idx = self.live_channel
        if idx >= subset.shape[1]:
            return
        if self._last_live_channel != idx:
            self.prepare_live_plot()
        plot = self.live_plot
        plot.clear()
        plot.plot(t, subset[:, idx], pen=pg.mkPen((59, 130, 246), width=2))
        plot.setTitle(f"Live CH {idx + 1}")
        plot.setLabel("bottom", "Time [s]")
        plot.setLabel("left", "Amplitude [V]")
        plot.setXRange(float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1 / self.freq, padding=0)

    def plot_frequency_domain(self) -> None:
        if self.data.size == 0:
            return
        n = len(self.data)
        freqs = (np.arange(n) - np.floor(n / 2)) * self.freq / n
        for idx, plot in enumerate(self.fourier_plots):
            plot.clear()
            if idx >= self.data.shape[1]:
                continue
            spectrum = fft_spectrum(self.data[:, idx], self.sample_idx)
            item = plot.plot(freqs, spectrum, pen=pg.mkPen((59, 130, 246), width=1))
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
            plot.setTitle(f"Ch {idx + 1}")

    def _on_tab_changed(self, index: int) -> None:
        if hasattr(self, "postproc_tab") and self.tabs.widget(index) is self.postproc_tab:
            self.update_postprocessing_plot()



    def load_postprocessing_csv(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Carica CSV acquisito",
            str(Path.cwd()),
            "CSV (*.csv);;Tutti i file (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("header CSV mancante")
                rows = list(reader)
            if not rows:
                raise ValueError("CSV vuoto")

            channels = [name for name in reader.fieldnames if name.upper().startswith("CH")]
            if not channels:
                channels = []
                for name in reader.fieldnames:
                    try:
                        float(rows[0][name])
                    except (TypeError, ValueError):
                        continue
                    channels.append(name)
            data = np.array([[float(row[name]) for name in channels] for row in rows], dtype=float)
        except Exception as exc:
            self.show_status(f"Caricamento CSV fallito: {exc}")
            return

        self._set_sample_data(data)
        self.sample_idx = len(data)
        self.postproc_channel_names = channels
        self.postproc_channel_combo.blockSignals(True)
        self.postproc_channel_combo.clear()
        self.postproc_channel_combo.addItems(channels)
        self.postproc_channel_combo.blockSignals(False)
        self.show_status(f"CSV caricato: {path.name} ({len(data)} campioni, {len(channels)} canali)")
        self.update_postprocessing_plot()

    def _postproc_params(self) -> dict[str, float | int | None]:
        return {
            "fs": float(self.postproc_fs_spin.value()) if hasattr(self, "postproc_fs_spin") else float(self.freq),
            "notch_hz": self.postproc_notch_spin.value() if self.postproc_notch_check.isChecked() else None,
            "notch_bw": self.postproc_notch_bw_spin.value(),
            "notch_q": self.postproc_notch_q_spin.value(),
            "hp": self.postproc_highpass_spin.value() if self.postproc_highpass_check.isChecked() else None,
            "lp": self.postproc_lowpass_spin.value() if self.postproc_lowpass_check.isChecked() else None,
            "fir_taps": int(self.postproc_fir_taps_spin.value()) | 1,
            "butter_order": int(self.postproc_butter_order_spin.value()),
            "frac_bits": int(self.postproc_frac_bits_spin.value()),
        }

    def _base_postproc_signal(self, raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = raw.astype(float).copy()
        steps: list[str] = []
        if self.postproc_offset_check.isChecked():
            x = x - np.mean(x)
            steps.append("offset")
        return x, steps

    def _postproc_fir_taps(self, params: dict[str, float | int | None]) -> list[np.ndarray]:
        if signal is None:
            return []
        fs = float(params["fs"])
        taps_n = int(params["fir_taps"])
        taps_list: list[np.ndarray] = []
        notch_hz = params["notch_hz"]
        hp = params["hp"]
        lp = params["lp"]
        if notch_hz is not None:
            taps_list.append(
                design_notch_50hz(fs, taps_n, center_hz=float(notch_hz), bw_hz=float(params["notch_bw"]))
            )
        if hp is not None and 0 < float(hp) < fs / 2:
            taps_list.append(design_highpass(fs, float(hp), taps_n))
        if lp is not None and 0 < float(lp) < fs / 2:
            if taps_n % 2 == 0:
                taps_n += 1
            taps_list.append(signal.firwin(taps_n, float(lp) / (fs / 2), pass_zero=True))
        return taps_list

    def _apply_fir_pipeline(
        self,
        x: np.ndarray,
        params: dict[str, float | int | None],
        realtime_fixed: bool,
    ) -> np.ndarray:
        taps_list = self._postproc_fir_taps(params)
        y = x.astype(float)
        if signal is None:
            return y
        for taps in taps_list:
            if realtime_fixed:
                taps_q, _ = quantize_taps(taps, int(params["frac_bits"]))
                y = fir_causal_fixed_point(y, taps_q, int(params["frac_bits"]))
            else:
                y = signal.filtfilt(taps, [1.0], y)
        return y

    def _postproc_butter_sos(self, params: dict[str, float | int | None], fixed: bool = False) -> np.ndarray | None:
        if signal is None:
            return None
        fs = float(params["fs"])
        sections: list[np.ndarray] = []
        notch_hz = params["notch_hz"]
        if notch_hz is not None and 0 < float(notch_hz) < fs / 2:
            b, a = signal.iirnotch(float(notch_hz), float(params["notch_q"]), fs=fs)
            sections.append(signal.tf2sos(b, a))
        hp = params["hp"]
        if hp is not None and 0 < float(hp) < fs / 2:
            sections.append(signal.butter(int(params["butter_order"]), float(hp), btype="highpass", fs=fs, output="sos"))
        lp = params["lp"]
        if lp is not None and 0 < float(lp) < fs / 2:
            sections.append(signal.butter(int(params["butter_order"]), float(lp), btype="lowpass", fs=fs, output="sos"))
        if not sections:
            return None
        sos = np.vstack(sections)
        if fixed:
            sos_q = quantize_sos(sos, int(params["frac_bits"]))
            sos = sos_q.astype(float) / (1 << int(params["frac_bits"]))
        return sos

    def _apply_butter_pipeline(
        self,
        x: np.ndarray,
        params: dict[str, float | int | None],
        mode: str,
    ) -> np.ndarray:
        fs = float(params["fs"])
        if mode == "offline":
            return apply_butterworth_offline_zero_phase(
                x,
                fs,
                hp_cutoff_hz=params["hp"],
                lp_cutoff_hz=params["lp"],
                order=int(params["butter_order"]),
                notch_hz=params["notch_hz"],
                notch_q=float(params["notch_q"]),
            )
        if mode == "realtime":
            return apply_butterworth_realtime_causal(
                x,
                fs,
                hp_cutoff_hz=params["hp"],
                lp_cutoff_hz=params["lp"],
                order=int(params["butter_order"]),
                notch_hz=params["notch_hz"],
                notch_q=float(params["notch_q"]),
            )
        return apply_butterworth_realtime_fixed_point(
            x,
            fs,
            hp_cutoff_hz=params["hp"],
            lp_cutoff_hz=params["lp"],
            order=int(params["butter_order"]),
            fractional_bits=int(params["frac_bits"]),
            notch_hz=params["notch_hz"],
            notch_q=float(params["notch_q"]),
        )

    def _selected_postproc_outputs(
        self,
        raw: np.ndarray,
        params: dict[str, float | int | None],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        base, steps = self._base_postproc_signal(raw)
        outputs: dict[str, np.ndarray] = {}
        selected = self._selected_postproc_filter_key()
        if selected == "fir_offline":
            outputs["FIR offline"] = self._apply_fir_pipeline(base, params, realtime_fixed=False)
        elif selected == "fir_realtime":
            outputs["FIR realtime fixed"] = self._apply_fir_pipeline(base, params, realtime_fixed=True)
        elif selected == "butter_offline":
            outputs["Butter offline"] = self._apply_butter_pipeline(base, params, "offline")
        elif selected == "butter_realtime":
            outputs["Butter realtime"] = self._apply_butter_pipeline(base, params, "realtime")
        elif selected == "butter_fixed":
            outputs["Butter fixed"] = self._apply_butter_pipeline(base, params, "fixed")
        if self.postproc_movavg_check.isChecked():
            for key in list(outputs):
                outputs[key] = moving_average(outputs[key], self.postproc_movavg_spin.value())
            steps.append(f"mov.avg {self.postproc_movavg_spin.value()}")
        return outputs, steps

    def process_channel(self, raw: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
        params = self._postproc_params()
        params["fs"] = float(fs)
        outputs, steps = self._selected_postproc_outputs(raw, params)
        if outputs:
            return next(iter(outputs.values())), steps
        base, base_steps = self._base_postproc_signal(raw)
        return base, base_steps

    def update_postprocessing_plot(self) -> None:
        if not hasattr(self, "postproc_plot"):
            return
        params = self._postproc_params()
        cache_signature = (
            self._data_revision,
            self.postproc_channel_combo.currentIndex(),
            self._selected_postproc_filter_key(),
            self.postproc_show_raw_check.isChecked(),
            self.postproc_offset_check.isChecked(),
            self.postproc_movavg_check.isChecked(),
            self.postproc_movavg_spin.value(),
            tuple(sorted(params.items())),
        )
        if cache_signature == self._postproc_cache_signature:
            return
        self.postproc_plot.clear()
        self.postproc_response_plot.clear()
        self.postproc_zplane_plot.clear()
        self.postproc_console.clear()
        if self.data.size == 0:
            self.postproc_plot.setTitle("Nessun dato registrato")
            return
        ch = self.postproc_channel_combo.currentIndex()
        if ch >= self.data.shape[1]:
            return
        raw = self.data[:, ch]
        fs = float(params["fs"])
        t = np.arange(len(raw)) / fs
        if self.postproc_show_raw_check.isChecked():
            item = self.postproc_plot.plot(t, raw, pen=pg.mkPen((150, 150, 150), width=1.3), name="raw")
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
        try:
            outputs, steps = self._selected_postproc_outputs(raw, params)
        except Exception as exc:
            self.postproc_console.setPlainText(f"Filtro fallito: {exc}")
            self.postproc_plot.setTitle("Filtro fallito")
            return
        colors = {
            "FIR offline": (37, 99, 235),
            "FIR realtime fixed": (20, 150, 90),
            "Butter offline": (220, 90, 70),
            "Butter realtime": (245, 158, 11),
            "Butter fixed": (147, 51, 234),
        }
        self.postproc_plot.addLegend()
        for name, values in outputs.items():
            item = self.postproc_plot.plot(
                t,
                values,
                pen=pg.mkPen(colors.get(name, (59, 130, 246)), width=2),
                name=name,
            )
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
        title = self.postproc_channel_combo.currentText() or f"CH{ch + 1}"
        if steps:
            title += " - " + ", ".join(steps)
        self.postproc_plot.setTitle(title)
        raw_for_range = raw if self.postproc_show_raw_check.isChecked() else np.array([])
        self._set_postproc_time_range(t, raw_for_range, list(outputs.values()))
        if self._selected_postproc_filter_key() == "none":
            self.postproc_response_plot.setTitle("Nessun filtro selezionato")
            self.postproc_zplane_plot.setTitle("Z-plane")
            self.postproc_console.setPlainText("Raw CH mostrato senza filtro. Seleziona un filtro dal menu per vedere parametri e diagnostica.")
        else:
            self._update_postproc_diagnostics(params)
        self._postproc_cache_signature = cache_signature

    def _set_postproc_time_range(self, t: np.ndarray, raw: np.ndarray, filtered: list[np.ndarray]) -> None:
        if len(t) == 0:
            return
        self.postproc_plot.setXRange(0, float(t[-1]) if len(t) > 1 else 1.0, padding=0)
        series = [raw.astype(float), *[values.astype(float) for values in filtered]]
        values = np.concatenate([item[np.isfinite(item)] for item in series if item.size])
        if values.size == 0:
            return
        low, high = np.percentile(values, [1, 99])
        if low == high:
            low, high = float(np.min(values)), float(np.max(values))
        margin = max((high - low) * 0.12, 1.0)
        self.postproc_plot.setYRange(float(low - margin), float(high + margin), padding=0)

    def _diagnostic_filter_key(self) -> str:
        return self._selected_postproc_filter_key()

    def _filter_zpk_response(
        self,
        params: dict[str, float | int | None],
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        if signal is None:
            raise RuntimeError("scipy non installato")
        fs = float(params["fs"])
        if key.startswith("fir"):
            taps_list = self._postproc_fir_taps(params)
            if not taps_list:
                taps = np.array([1.0])
            else:
                taps = taps_list[0]
                for extra in taps_list[1:]:
                    taps = np.convolve(taps, extra)
            if key == "fir_realtime":
                _, taps = quantize_taps(taps, int(params["frac_bits"]))
            freq, response = signal.freqz(taps, [1.0], worN=4096, fs=fs)
            zeros = np.roots(taps) if len(taps) > 1 else np.array([])
            poles = np.array([])
            label = f"{self.postproc_diagnostic_combo.currentText()} | FIR ordine {len(taps) - 1}"
            return freq, response, zeros, poles, label

        sos = self._postproc_butter_sos(params, fixed=(key == "butter_fixed"))
        if sos is None:
            sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        freq, response = signal.sosfreqz(sos, worN=4096, fs=fs)
        zeros, poles, gain = signal.sos2zpk(sos)
        label = f"{self.postproc_diagnostic_combo.currentText()} | SOS={len(sos)} gain={gain:.4g}"
        return freq, response, zeros, poles, label

    def _update_postproc_diagnostics(self, params: dict[str, float | int | None]) -> None:
        if signal is None:
            return
        try:
            key = self._diagnostic_filter_key()
            freq, response, zeros, poles, label = self._filter_zpk_response(params, key)
        except Exception as exc:
            self.postproc_console.setPlainText(f"Diagnostica filtro fallita: {exc}")
            return

        mag_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
        self.postproc_response_plot.plot(freq, mag_db, pen=pg.mkPen((37, 99, 235), width=2))
        self.postproc_response_plot.setTitle(label)
        self.postproc_response_plot.setXRange(0, min(float(params["fs"]) / 2, 250), padding=0)

        theta = np.linspace(0, 2 * np.pi, 256)
        self.postproc_zplane_plot.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen((170, 170, 170), width=1))
        if len(zeros):
            self.postproc_zplane_plot.plot(
                np.real(zeros),
                np.imag(zeros),
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=(37, 99, 235),
                name="zeros",
            )
        if len(poles):
            self.postproc_zplane_plot.plot(
                np.real(poles),
                np.imag(poles),
                pen=None,
                symbol="x",
                symbolSize=9,
                symbolPen=pg.mkPen((220, 90, 70), width=2),
                name="poles",
            )
        self.postproc_zplane_plot.setTitle("Z-plane")
        self.postproc_zplane_plot.setXRange(-1.2, 1.2, padding=0)
        self.postproc_zplane_plot.setYRange(-1.2, 1.2, padding=0)

        lines = [
            label,
            f"Fs={float(params['fs']):g} Hz",
            f"notch={params['notch_hz'] if params['notch_hz'] is not None else 'off'} Hz",
            f"HP={params['hp'] if params['hp'] is not None else 'off'} Hz, LP={params['lp'] if params['lp'] is not None else 'off'} Hz",
            f"FIR taps={int(params['fir_taps'])}, Butter order={int(params['butter_order'])}, frac bits={int(params['frac_bits'])}",
            f"zeros={len(zeros)}, poles={len(poles)}",
        ]
        if len(poles):
            lines.append(f"max |pole|={np.max(np.abs(poles)):.6f}")
        if len(zeros):
            lines.append(f"max |zero|={np.max(np.abs(zeros)):.6f}")
        if key == "butter_fixed":
            lines.extend(self._postproc_fixed_accumulator_lines(params))
        lines.extend(self._postproc_equation_lines(params, key))
        self.postproc_console.setPlainText("\n".join(lines))

    def _postproc_fixed_accumulator_lines(self, params: dict[str, float | int | None]) -> list[str]:
        ch = self.postproc_channel_combo.currentIndex()
        if self.data.size == 0 or ch >= self.data.shape[1]:
            return []
        sos_float = self._postproc_butter_sos(params, fixed=False)
        if sos_float is None:
            return []
        raw = self.data[:, ch]
        base, _ = self._base_postproc_signal(raw)
        frac_bits = int(params["frac_bits"])
        sos_q = quantize_sos(sos_float, frac_bits)
        _y, section_stats = sos_fixed_point_section_stats(base, sos_q, frac_bits)
        lines = ["", "Range accumulatore misurato sul canale corrente:"]
        for item in section_stats:
            lines.append(
                f"SOS{item['section']}: max|acc|={item['max_abs_acc']} "
                f"-> acc_bits={item['required_acc_bits']} | "
                f"max|x_state|={item['max_abs_x_state']} ({item['required_x_bits']} bit) | "
                f"max|y_state|={item['max_abs_y_state']} ({item['required_y_bits']} bit)"
            )
        max_bits = max((item["required_acc_bits"] for item in section_stats), default=1)
        lines.append(f"Consiglio pratico: accumulatore signed almeno {max_bits + 4} bit (4 bit di margine).")
        lines.append("In RTL usa saturazione sull'output/stati, non wrap-around.")
        return lines

    def _postproc_equation_lines(self, params: dict[str, float | int | None], key: str) -> list[str]:
        lines = ["", "Equazione:"]
        if key.startswith("fir"):
            taps_list = self._postproc_fir_taps(params)
            if not taps_list:
                return [*lines, "y[n] = x[n]"]
            taps = taps_list[0]
            for extra in taps_list[1:]:
                taps = np.convolve(taps, extra)
            lines.append("y[n] = sum_k b[k] * x[n-k]")
            lines.append(f"numero coefficienti equivalenti={len(taps)}")
            lines.append("primi coefficienti b:")
            preview = taps[: min(8, len(taps))]
            lines.append(", ".join(f"{value:.6g}" for value in preview))
            if key == "fir_realtime":
                taps_q, _ = quantize_taps(taps, int(params["frac_bits"]))
                lines.append(f"primi coefficienti interi Q*.{int(params['frac_bits'])}:")
                lines.append(", ".join(str(int(value)) for value in taps_q[: min(8, len(taps_q))]))
            return lines

        fixed = key == "butter_fixed"
        sos_float = self._postproc_butter_sos(params, fixed=False)
        sos_used = self._postproc_butter_sos(params, fixed=fixed)
        if sos_used is None:
            return [*lines, "y[n] = x[n]"]
        lines.append("Per ogni biquad:")
        lines.append("y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]")
        lines.append("con a0 normalizzato a 1.")
        lines.append("")
        lines.append("Coefficienti usati [b0 b1 b2 a0 a1 a2]:")
        for idx, section in enumerate(sos_used, start=1):
            lines.append(f"SOS{idx}: " + ", ".join(f"{value:.8g}" for value in section))
        if fixed and sos_float is not None:
            sos_q = quantize_sos(sos_float, int(params["frac_bits"]))
            lines.append("")
            lines.append(f"Coefficienti interi Q*.{int(params['frac_bits'])}:")
            for idx, section in enumerate(sos_q, start=1):
                lines.append(f"SOS{idx}: " + ", ".join(str(int(value)) for value in section))
            lines.append("")
            lines.append("Implementazione fixed-point:")
            lines.append("acc = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2")
            lines.append(f"y0 = acc >> {int(params['frac_bits'])}")
        return lines

    def panel_for_ads(self, ads_name: str) -> RegisterPanel:
        return self.ads1_panel if ads_name == "ADS1" else self.ads2_panel

    @QtCore.Slot(bool, str)
    def set_connected(self, connected: bool, port: str) -> None:
        self.connected = connected
        self._set_lamp(connected)
        if connected:
            self.show_status(f"Connesso a {port} @ {BAUDRATE}.")

    def _set_lamp(self, connected: bool) -> None:
        color = "#00cc3a" if connected else "#ff2020"
        self.connection_lamp.setStyleSheet(
            f"border-radius: 17px; background: {color}; border: 2px solid #aa0000;"
        )

    def log_serial(self, line: str) -> None:
        self.log_lines.append(line)
        if hasattr(self, "serial_log"):
            self.serial_log.appendPlainText(line)

    @QtCore.Slot(str)
    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        if hasattr(self, "serial_log"):
            self.serial_log.appendPlainText(f"# {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.close_requested.emit()
        self.live_inference_enabled = False
        self.inference_thread.quit()
        self.inference_thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOptions(antialias=True)
    qss_path = Path(__file__).parent / "resources" / "eeg_emg_clinical.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())
    window = EEGEMGApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
