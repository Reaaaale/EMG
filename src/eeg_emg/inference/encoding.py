from __future__ import annotations

import numpy as np


def expand_window_with_derivatives(window: np.ndarray, max_derivative_order: int = 0) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if max_derivative_order == 0:
        return window

    time_steps, old_dim_size = window.shape
    new_dim_size = old_dim_size * (max_derivative_order + 1)
    expanded = np.zeros((time_steps, new_dim_size), dtype=np.float32)
    expanded[:, :old_dim_size] = window

    for n in range(1, max_derivative_order + 1):
        prev_start = old_dim_size * (n - 1)
        curr_start = old_dim_size * n
        count = time_steps - n * 4
        if count <= 0:
            continue
       
        previous = expanded[:, prev_start : prev_start + old_dim_size].astype(
            np.float64,
            copy=False,
        )
        expanded[
            n * 2 : n * 2 + count,
            curr_start : curr_start + old_dim_size,
        ] = (
            -previous[:count]
            - 2 * previous[1 : count + 1]
            + 2 * previous[2 : count + 2]
            + previous[3 : count + 3]
        )

    trim = max_derivative_order * 2
    return expanded[trim:-trim, :]


def make_delta_vector(
    n_features: int,
    original_channels: int,
    delta_by_order: float | list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    values = np.atleast_1d(np.asarray(delta_by_order, dtype=np.float32))
    required_orders = (n_features + original_channels - 1) // original_channels
    if len(values) < required_orders:
        raise ValueError(
            "delta_by_order deve contenere un valore per ogni ordine di derivata"
        )
    return np.repeat(values[:required_orders], original_channels)[:n_features]


def delta_encode_standard(expanded_window: np.ndarray, delta_values: np.ndarray) -> np.ndarray:
    time_steps, features = expanded_window.shape
    spikes = np.zeros((features * 2, time_steps), dtype=np.float32)
    ref = expanded_window[0].copy()

    for t in range(time_steps):
        diff = expanded_window[t] - ref
        pos = diff > delta_values
        neg = diff < -delta_values
        spikes[0::2, t] = pos.astype(np.float32)
        spikes[1::2, t] = neg.astype(np.float32)
        fired = pos | neg
        ref[fired] = expanded_window[t, fired]

    return spikes


def delta_encode_lambda(
    expanded_window: np.ndarray,
    delta_values: np.ndarray,
    lambda_d: float = 0.001,
) -> np.ndarray:
    """Lambda delta encoding, equivalent to the training notebook."""
    if not 0.0 <= lambda_d < 1.0:
        raise ValueError("lambda_d deve essere compreso tra 0 incluso e 1 escluso")

    time_steps, features = expanded_window.shape
    spikes = np.zeros((features * 2, time_steps), dtype=np.float32)
    ref = expanded_window[0].copy()

    for t in range(1, time_steps):
        ref = ref * (1.0 - lambda_d)
        diff = expanded_window[t] - ref
        pos = diff >= delta_values
        neg = diff <= -delta_values

        spikes[0::2, t] = pos.astype(np.float32)
        spikes[1::2, t] = neg.astype(np.float32)
        ref[pos] = ref[pos] + delta_values[pos]
        ref[neg] = ref[neg] - delta_values[neg]

    return spikes


def encode_window(
    window: np.ndarray,
    delta: float,
    max_derivative_order: int = 0,
    mode: str = "standard",
    lambda_d: float = 0.001,
    delta_by_order: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    original_channels = window.shape[1]
    expanded = expand_window_with_derivatives(window, max_derivative_order=max_derivative_order)
    if delta_by_order is None:
        delta_by_order = [delta] * (max_derivative_order + 1)
    delta_values = make_delta_vector(
        expanded.shape[1],
        original_channels,
        delta_by_order,
    )

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "standard":
        return delta_encode_standard(expanded, delta_values)
    if normalized_mode == "lambda":
        return delta_encode_lambda(expanded, delta_values, lambda_d=lambda_d)
    raise ValueError("mode deve essere 'standard' oppure 'lambda'")
