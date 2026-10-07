from __future__ import annotations

from pathlib import Path


def read_training_config(model_path: str | Path) -> dict:
    import json

    path = Path(model_path)
    config_path = path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def config_input_features(config: dict, fallback: int = 16) -> int:
    value = config.get("input_features")
    if value is None:
        return fallback
    return int(value)


def config_num_classes(config: dict, fallback: int = 4) -> int:
    value = config.get("num_classes")
    if value is None:
        return fallback
    return int(value)


def config_max_derivative_order(config: dict, n_channels: int = 8) -> int:
    input_features = config_input_features(config, fallback=n_channels * 2)
    return max(0, input_features // (n_channels * 2) - 1)


class NetworkBuilder:
    def __init__(self, input_features: int, num_classes: int = 4, dropout_p: float = 0.05) -> None:
        self.input_features = int(input_features)
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout_p)

    def build(self):
        import torch
        import lava.lib.dl.slayer as slayer

        class Network(torch.nn.Module):
            def __init__(self, input_features: int, num_classes: int, dropout_p: float) -> None:
                super().__init__()
                neuron_params = {
                    "threshold": 1.25,
                    "current_decay": 1,
                    "voltage_decay": 0.03,
                    "tau_grad": 0.03,
                    "scale_grad": 3,
                    "requires_grad": True,
                }
                neuron_params_drop = {
                    **neuron_params,
                    "dropout": slayer.neuron.Dropout(p=dropout_p),
                }
                self.blocks = torch.nn.ModuleList([
                    slayer.block.cuba.Dense(neuron_params_drop, input_features, 64, weight_norm=True, delay=True),
                    slayer.block.cuba.Dense(neuron_params_drop, 64, 128, weight_norm=True, delay=True),
                    slayer.block.cuba.Dense(neuron_params, 128, num_classes, weight_norm=True),
                ])

            def forward(self, spike):
                for block in self.blocks:
                    spike = block(spike)
                return spike

        return Network(self.input_features, self.num_classes, self.dropout_p)


try:
    import torch

    _BaseNetwork = torch.nn.Module
except Exception:  
    _BaseNetwork = object


class Network(_BaseNetwork):
    """Compatibility class for full models saved from a notebook __main__.Network."""

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike
