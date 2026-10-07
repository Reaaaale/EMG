from __future__ import annotations

import threading

from PySide6 import QtCore

from .config import BAUDRATE

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


class SerialWorker(QtCore.QObject):
    line_received = QtCore.Signal(str)
    connected = QtCore.Signal(bool, str)
    error = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._serial = None
        self._running = False
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    @QtCore.Slot(str)
    def open(self, port: str) -> None:
        if serial is None:
            self.error.emit("pyserial non e installato.")
            self.connected.emit(False, "")
            return
        try:
            self.close()
            with self._lock:
                self._serial = serial.Serial(port, BAUDRATE, timeout=0.1)
                self._running = True
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            self.connected.emit(True, port)
        except Exception as exc:
            self.error.emit(f"Connessione fallita: {exc}")
            self.connected.emit(False, "")

    @QtCore.Slot(str)
    def write_line(self, line: str) -> None:
        with self._lock:
            if not self._serial or not self._serial.is_open:
                self.error.emit(f"Seriale non connessa: impossibile inviare '{line}'.")
                return
            try:
                self._serial.write((line + "\n").encode())
            except Exception as exc:
                self.error.emit(f"Errore scrittura seriale: {exc}")

    @QtCore.Slot()
    def close(self) -> None:
        self._running = False
        with self._lock:
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
        self.connected.emit(False, "")

    def _read_loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    port = self._serial
                if not port or not port.is_open:
                    break
                raw = port.readline()
            except Exception as exc:
                if self._running:
                    self.error.emit(f"Errore lettura seriale: {exc}")
                break
            if raw:
                self.line_received.emit(raw.decode(errors="replace").strip())
        self.close()


