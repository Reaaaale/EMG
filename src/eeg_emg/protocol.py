from __future__ import annotations

import re

import numpy as np

from .config import CHANNEL_COUNT

REGISTER_RE = re.compile(r"(ADS[12]):\s*REG\[0x([0-9A-Fa-f]+)\]\s*=\s*0x([0-9A-Fa-f]+)")


def parse_register_line(line: str) -> tuple[str, int, int] | None:
    match = REGISTER_RE.match(line)
    if not match:
        return None
    return match.group(1), int(match.group(2), 16), int(match.group(3), 16)


def parse_channel_line(line: str, expected: int, scale: bool) -> np.ndarray:
    _, payload = line.split(":", 1)
    values = [float(part.strip()) for part in payload.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"Attesi {expected} canali, ricevuti {len(values)}.")
    arr = np.zeros(CHANNEL_COUNT, dtype=float)
    arr[:expected] = values
    if scale:
        arr[:expected] *= 5 / (2**24)
    return arr


def read_registers_command(ads_name: str) -> str:
    return "READREG1" if ads_name == "ADS1" else "READREG2"


def gain_command(ads_name: str, channel: int, gain: int) -> str:
    return f"GAIN {ads_name} {channel} {gain}"


def input_command(ads_name: str, channel: int, input_mode: str) -> str:
    return f"INPUT {ads_name} {channel} {input_mode}"


def unipolar_command(ads_name: str, enabled: bool) -> str:
    return f"SRB1{'ON' if enabled else 'OFF'}: {ads_name}"


def ads2_enabled_command(enabled: bool) -> str:
    return "ADS2ON" if enabled else "ADSOFF"


def frequency_command(freq: int) -> str:
    return f"FREQ {freq}"
