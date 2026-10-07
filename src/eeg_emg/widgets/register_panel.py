from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtWidgets

from ..config import INPUT_MODES, PGA_VALUES, REGISTER_NAMES


class RegisterPanel(QtWidgets.QGroupBox):
    read_all_requested = QtCore.Signal(str)
    update_channel_requested = QtCore.Signal(str)
    update_all_requested = QtCore.Signal(str)
    unipolar_changed = QtCore.Signal(str, bool)

    def __init__(self, ads_name: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(ads_name, parent)
        self.ads_name = ads_name
        self.selected_channel = 1
        self.pga = [1] * 8
        self.channel_on = [1] * 8
        self.input_mode = "TEST"
        self.reg_fields: dict[int, QtWidgets.QLineEdit] = {}
        self.noise_fields: list[QtWidgets.QLineEdit] = []
        self._build()

    def _build(self) -> None:
        main = QtWidgets.QGridLayout(self)
        main.setHorizontalSpacing(10)
        main.setVerticalSpacing(4)

        for idx, name in enumerate(REGISTER_NAMES):
            col = 0 if idx < 12 else 2
            row = idx if idx < 12 else idx - 12
            label = QtWidgets.QLabel(name)
            field = QtWidgets.QLineEdit()
            field.setMaxLength(4)
            field.setPlaceholderText("--")
            field.setReadOnly(True)
            field.setFixedWidth(86)
            main.addWidget(label, row, col)
            main.addWidget(field, row, col + 1)
            self.reg_fields[idx] = field

        noise_box = QtWidgets.QGroupBox("NOISE")
        noise_layout = QtWidgets.QGridLayout(noise_box)
        for idx in range(8):
            noise_layout.addWidget(QtWidgets.QLabel(f"CH{idx + 1}"), idx, 0)
            field = QtWidgets.QLineEdit()
            field.setReadOnly(True)
            self.noise_fields.append(field)
            noise_layout.addWidget(field, idx, 1)
        main.addWidget(noise_box, 0, 4, 10, 2)

        controls = QtWidgets.QGridLayout()
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems([str(i) for i in range(1, 9)])
        self.channel_combo.currentTextChanged.connect(self._channel_changed)
        self.on_combo = QtWidgets.QComboBox()
        self.on_combo.addItems(["1", "0"])
        self.on_combo.currentTextChanged.connect(self._on_changed)
        self.pga_combo = QtWidgets.QComboBox()
        self.pga_combo.addItems(PGA_VALUES)
        self.pga_combo.currentTextChanged.connect(self._pga_changed)
        self.input_combo = QtWidgets.QComboBox()
        self.input_combo.addItems(INPUT_MODES)
        self.input_combo.setCurrentText("TEST")
        self.input_combo.currentTextChanged.connect(self._input_changed)

        controls.addWidget(QtWidgets.QLabel("CH"), 0, 0)
        controls.addWidget(self.channel_combo, 0, 1)
        controls.addWidget(QtWidgets.QLabel("ON"), 0, 2)
        controls.addWidget(self.on_combo, 0, 3)
        controls.addWidget(QtWidgets.QLabel("PGA"), 0, 4)
        controls.addWidget(self.pga_combo, 0, 5)
        controls.addWidget(QtWidgets.QLabel("In"), 0, 6)
        controls.addWidget(self.input_combo, 0, 7)

        read_button = QtWidgets.QPushButton("Read ALL REGS")
        update_ch_button = QtWidgets.QPushButton("Update CH")
        update_all_button = QtWidgets.QPushButton("Update ALL")
        self.unipolar_check = QtWidgets.QCheckBox("Unipolar")

        read_button.clicked.connect(lambda: self.read_all_requested.emit(self.ads_name))
        update_ch_button.clicked.connect(lambda: self.update_channel_requested.emit(self.ads_name))
        update_all_button.clicked.connect(lambda: self.update_all_requested.emit(self.ads_name))
        self.unipolar_check.toggled.connect(
            lambda checked: self.unipolar_changed.emit(self.ads_name, checked)
        )

        controls.addWidget(read_button, 1, 1, 1, 2)
        controls.addWidget(update_ch_button, 1, 3, 1, 2)
        controls.addWidget(update_all_button, 1, 5, 1, 2)
        controls.addWidget(self.unipolar_check, 1, 7)
        main.addLayout(controls, 12, 0, 2, 6)

    def _channel_changed(self, value: str) -> None:
        self.selected_channel = int(value)
        idx = self.selected_channel - 1
        self.on_combo.setCurrentText(str(self.channel_on[idx]))
        pga_value = str(self.pga[idx])
        if pga_value in PGA_VALUES:
            self.pga_combo.setCurrentText(pga_value)

    def _on_changed(self, value: str) -> None:
        self.channel_on[self.selected_channel - 1] = int(value)

    def _pga_changed(self, value: str) -> None:
        if value.isdigit():
            self.pga[self.selected_channel - 1] = int(value)

    def _input_changed(self, value: str) -> None:
        self.input_mode = value

    def set_register(self, address: int, value: int) -> None:
        field = self.reg_fields.get(address)
        if field:
            field.setText(f"{value:02X}")

    def set_noise(self, values: Iterable[float]) -> None:
        for field, value in zip(self.noise_fields, values):
            field.setText(f"{value:.6g}")


