from __future__ import annotations

import numpy as np

try:
    from scipy import signal
except ImportError:
    signal = None


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def odd_taps(x_len: int, requested: int = 201) -> int:
    taps = min(requested, x_len - 1 if x_len % 2 == 0 else x_len)
    if taps % 2 == 0:
        taps -= 1
    return taps


def fft_spectrum(x: np.ndarray, sample_count: int) -> np.ndarray:
    centered = x - np.mean(x)
    return np.abs(np.fft.fftshift(np.fft.fft(centered))) / max(sample_count, 1)


def design_notch_50hz(fs: float, numtaps: int = 201, center_hz: float = 50, bw_hz: float = 2) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    if numtaps % 2 == 0:
        numtaps += 1
    nyq = fs / 2
    bands = [
        0,
        (center_hz - bw_hz) / nyq,
        (center_hz - 0.5) / nyq,
        (center_hz + 0.5) / nyq,
        (center_hz + bw_hz) / nyq,
        1,
    ]
    return signal.firls(numtaps, bands, [1, 1, 0, 0, 1, 1])


def design_highpass(fs: float, cutoff_hz: float, numtaps: int = 201) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    if numtaps % 2 == 0:
        numtaps += 1
    return signal.firwin(numtaps, cutoff_hz / (fs / 2), pass_zero=False)


def design_butterworth(
    fs: float,
    cutoff_hz: float,
    kind: str,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if signal is None:
        raise RuntimeError("scipy non installato")
    nyq = fs / 2
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff_hz deve essere tra 0 e Nyquist ({nyq:g} Hz)")
    if order < 1:
        raise ValueError("order deve essere >= 1")
    if kind not in {"highpass", "lowpass"}:
        raise ValueError("kind deve essere 'highpass' oppure 'lowpass'")
    return signal.butter(order, cutoff_hz / nyq, btype=kind)


def apply_fir_offline_zero_phase(
    x: np.ndarray,
    fs: float,
    hp_cutoff_hz: float = 5,
    numtaps: int = 51,
    notch_hz: float = 50,
    notch_bw_hz: float = 2,
) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    notch = design_notch_50hz(fs, numtaps, center_hz=notch_hz, bw_hz=notch_bw_hz)
    hp = design_highpass(fs, hp_cutoff_hz, numtaps)
    y = signal.filtfilt(notch, [1.0], x.astype(float))
    y = signal.filtfilt(hp, [1.0], y)
    return y


def apply_fir_realtime_fixed_point(
    x: np.ndarray,
    fs: float,
    hp_cutoff_hz: float = 5,
    numtaps: int = 51,
    fractional_bits: int = 15,
    notch_hz: float = 50,
    notch_bw_hz: float = 2,
) -> np.ndarray:
    notch = design_notch_50hz(fs, numtaps, center_hz=notch_hz, bw_hz=notch_bw_hz)
    hp = design_highpass(fs, hp_cutoff_hz, numtaps)
    notch_q, _ = quantize_taps(notch, fractional_bits)
    hp_q, _ = quantize_taps(hp, fractional_bits)
    y = fir_causal_fixed_point(x, notch_q, fractional_bits)
    y = fir_causal_fixed_point(y, hp_q, fractional_bits)
    return y


def apply_butterworth_offline_zero_phase(
    x: np.ndarray,
    fs: float,
    hp_cutoff_hz: float | None = 5,
    lp_cutoff_hz: float | None = None,
    order: int = 4,
    notch_hz: float | None = 50,
    notch_q: float = 30,
) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    y = x.astype(float)
    nyq = fs / 2

    if notch_hz is not None and 0 < notch_hz < nyq:
        b_notch, a_notch = signal.iirnotch(notch_hz, notch_q, fs=fs)
        y = signal.filtfilt(b_notch, a_notch, y)

    if hp_cutoff_hz is not None:
        b_hp, a_hp = design_butterworth(fs, hp_cutoff_hz, "highpass", order=order)
        y = signal.filtfilt(b_hp, a_hp, y)

    if lp_cutoff_hz is not None:
        b_lp, a_lp = design_butterworth(fs, lp_cutoff_hz, "lowpass", order=order)
        y = signal.filtfilt(b_lp, a_lp, y)

    return y


def apply_butterworth_realtime_causal(
    x: np.ndarray,
    fs: float,
    hp_cutoff_hz: float | None = 5,
    lp_cutoff_hz: float | None = None,
    order: int = 4,
    notch_hz: float | None = 50,
    notch_q: float = 30,
) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    y = x.astype(float)
    nyq = fs / 2

    if notch_hz is not None and 0 < notch_hz < nyq:
        b_notch, a_notch = signal.iirnotch(notch_hz, notch_q, fs=fs)
        sos_notch = signal.tf2sos(b_notch, a_notch)
        y = signal.sosfilt(sos_notch, y)

    if hp_cutoff_hz is not None:
        sos_hp = signal.butter(order, hp_cutoff_hz, btype="highpass", fs=fs, output="sos")
        y = signal.sosfilt(sos_hp, y)

    if lp_cutoff_hz is not None:
        sos_lp = signal.butter(order, lp_cutoff_hz, btype="lowpass", fs=fs, output="sos")
        y = signal.sosfilt(sos_lp, y)

    return y


def quantize_coefficients(sos_float: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    scale = 1 << fractional_bits
    quantized_sections: list[list[int]] = []

    for section in sos_float:
        b0, b1, b2, a0, a1, a2 = section.astype(float)

        # Normalize the section so the hardware equation uses a0 = 1.
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        quantized_sections.append(
            [
                round(b0 * scale),
                round(b1 * scale),
                round(b2 * scale),
                scale,  # a0 = 1 in Q format
                round(a1 * scale),
                round(a2 * scale),
            ]
        )

    return np.asarray(quantized_sections, dtype=np.int64)


def biquad_filter_fixed(x: np.ndarray, coefficients_q: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    b0, b1, b2, _a0, a1, a2 = [int(value) for value in coefficients_q]

    output = np.zeros(len(x), dtype=np.int64)
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0

    for idx, sample in enumerate(x):
        x0 = int(round(sample))

        acc = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y0 = acc >> fractional_bits

        output[idx] = y0

        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0

    return output


def sos_filter_fixed(x: np.ndarray, sos_q: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    y = np.asarray(x, dtype=np.int64)

    for coefficients_q in sos_q:
        y = biquad_filter_fixed(y, coefficients_q, fractional_bits)

    return y.astype(float)


def quantize_sos(sos: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    return quantize_coefficients(sos, fractional_bits)


def biquad_fixed_point(x: np.ndarray, coeff_q: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    return biquad_filter_fixed(x, coeff_q, fractional_bits)


def sos_causal_fixed_point(x: np.ndarray, sos_q: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    return sos_filter_fixed(np.round(x).astype(np.int64), sos_q, fractional_bits)


def design_butterworth_pipeline_sos(
    fs: float,
    hp_cutoff_hz: float | None = 5,
    lp_cutoff_hz: float | None = None,
    order: int = 4,
    notch_hz: float | None = 50,
    notch_q: float = 30,
) -> np.ndarray | None:
    if signal is None:
        raise RuntimeError("scipy non installato")

    sections: list[np.ndarray] = []
    nyq = fs / 2

    if notch_hz is not None and 0 < notch_hz < nyq:
        b_notch, a_notch = signal.iirnotch(notch_hz, notch_q, fs=fs)
        sections.append(signal.tf2sos(b_notch, a_notch))

    if hp_cutoff_hz is not None:
        sections.append(signal.butter(order, hp_cutoff_hz, btype="highpass", fs=fs, output="sos"))

    if lp_cutoff_hz is not None:
        sections.append(signal.butter(order, lp_cutoff_hz, btype="lowpass", fs=fs, output="sos"))

    if not sections:
        return None

    return np.vstack(sections)


class StreamingButterworthFilter:
    """Stateful causal Butterworth pipeline for multichannel live samples.

    The coefficients match :func:`apply_butterworth_realtime_causal`, while
    ``zi`` is retained between calls so processing one sample at a time gives
    the same result as filtering the complete recording in one call.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        hp_cutoff_hz: float | None = 5,
        lp_cutoff_hz: float | None = None,
        order: int = 4,
        notch_hz: float | None = 50,
        notch_q: float = 30,
    ) -> None:
        if signal is None:
            raise RuntimeError("scipy non installato")
        self.n_channels = int(n_channels)
        self.sos = design_butterworth_pipeline_sos(
            fs=fs,
            hp_cutoff_hz=hp_cutoff_hz,
            lp_cutoff_hz=lp_cutoff_hz,
            order=order,
            notch_hz=notch_hz,
            notch_q=notch_q,
        )
        self.reset()

    def reset(self) -> None:
        section_count = 0 if self.sos is None else self.sos.shape[0]
        self._zi = np.zeros((section_count, 2, self.n_channels), dtype=float)

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        values = np.asarray(sample, dtype=float)
        if values.shape != (self.n_channels,):
            raise ValueError(
                f"attesi {self.n_channels} canali, ricevuta shape {values.shape}"
            )
        if self.sos is None:
            return values.copy()

        filtered, self._zi = signal.sosfilt(
            self.sos,
            values[np.newaxis, :],
            axis=0,
            zi=self._zi,
        )
        return filtered[0]


def required_signed_bits(value: int) -> int:
    magnitude = abs(int(value))
    if magnitude == 0:
        return 1
    return int(np.ceil(np.log2(magnitude + 1))) + 1


def sos_fixed_point_section_stats(
    x: np.ndarray,
    sos_q: np.ndarray,
    fractional_bits: int = 15,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    y = [int(round(value)) for value in x]
    stats: list[dict[str, int]] = []

    for section_index, section in enumerate(sos_q, start=1):
        b0, b1, b2, _a0, a1, a2 = [int(value) for value in section]
        out: list[int] = []
        x1 = x2 = y1 = y2 = 0
        max_abs_acc = 0
        max_abs_x = 0
        max_abs_y = 0

        for x0 in y:
            acc = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            yn = acc >> fractional_bits
            out.append(yn)
            max_abs_acc = max(max_abs_acc, abs(acc))
            max_abs_x = max(max_abs_x, abs(x0), abs(x1), abs(x2))
            max_abs_y = max(max_abs_y, abs(yn), abs(y1), abs(y2))
            x2 = x1
            x1 = x0
            y2 = y1
            y1 = yn

        stats.append(
            {
                "section": section_index,
                "max_abs_acc": max_abs_acc,
                "required_acc_bits": required_signed_bits(max_abs_acc),
                "max_abs_x_state": max_abs_x,
                "required_x_bits": required_signed_bits(max_abs_x),
                "max_abs_y_state": max_abs_y,
                "required_y_bits": required_signed_bits(max_abs_y),
            }
        )
        y = out

    return np.asarray(y, dtype=float), stats


def apply_butterworth_realtime_fixed_point(
    x: np.ndarray,
    fs: float,
    hp_cutoff_hz: float | None = 5,
    lp_cutoff_hz: float | None = None,
    order: int = 4,
    fractional_bits: int = 15,
    notch_hz: float | None = 50,
    notch_q: float = 30,
) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")

    # 1. Design offline dei coefficienti in formato SOS.
    coefficients_float = design_butterworth_pipeline_sos(
        fs=fs,
        hp_cutoff_hz=hp_cutoff_hz,
        lp_cutoff_hz=lp_cutoff_hz,
        order=order,
        notch_hz=notch_hz,
        notch_q=notch_q,
    )
    if coefficients_float is None:
        return x.astype(float)

    # 2. Quantizzazione fixed-point dei coefficienti, per esempio Q*.15.
    coefficients_fixed = quantize_coefficients(coefficients_float, fractional_bits)

    # 3. Applicazione causale sample-by-sample della cascata di biquad.
    filtered = sos_filter_fixed(x, coefficients_fixed, fractional_bits)
    return filtered


def frequency_response(taps: np.ndarray, fs: float, wor_n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    if signal is None:
        raise RuntimeError("scipy non installato")
    freq, response = signal.freqz(taps, worN=wor_n, fs=fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
    return freq, magnitude_db


def quantize_taps(taps: np.ndarray, fractional_bits: int = 15) -> tuple[np.ndarray, np.ndarray]:
    scale = 1 << fractional_bits
    quantized = np.round(taps * scale).astype(np.int64)
    restored = quantized.astype(float) / scale
    return quantized, restored


def fir_causal_float(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    if signal is None:
        raise RuntimeError("scipy non installato")
    return signal.lfilter(taps, [1.0], x.astype(float))


def fir_causal_fixed_point(x: np.ndarray, taps_q: np.ndarray, fractional_bits: int = 15) -> np.ndarray:
    x_i = np.round(x).astype(np.int64)
    taps_i = taps_q.astype(np.int64)
    y = np.zeros(len(x_i), dtype=np.int64)

    for n in range(len(x_i)):
        acc = np.int64(0)
        max_k = min(n + 1, len(taps_i))
        for k in range(max_k):
            acc += taps_i[k] * x_i[n - k]
        y[n] = acc >> fractional_bits
    return y.astype(float)


def group_delay_samples(numtaps: int) -> int:
    return (numtaps - 1) // 2
