from math import pi
from torch.nn import Module
from torch.nn.modules.activation import (
    CELU,
    ELU,
    Hardsigmoid,
    LogSigmoid,
    ReLU6,
    SELU,
    Sigmoid,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
    Hardtanh
)

_max = float("+inf")
_min = float("-inf")

_min_max = (_min, _max)

_ACTIVATION_THRESHOLDS= {
    CELU : (-1.0, _max),
    ELU : (-1.0, _max),
    Hardsigmoid : (0.0, 1.0),
    LogSigmoid :(_min, 0.0),
    ReLU6 : (0.0, 6.0),
    SELU : (-1.7580993408473766, _max), # scale * alpha * (-1)
    Sigmoid : (0.0, 1.0),
    Softsign : (-1.0, 1.0),
    Tanh : (-1.0, 1.0),
    Tanhshrink : _min_max,
}

_ACTIVATION_THRESHOLD_RUNTIME = {
    Threshold : lambda module: (module.value, _max),
    Hardtanh  : lambda module: (module.min_val, module.max_val)
}

def threshold_for_module(module : Module|None) -> tuple[float, float]:
    if module.__class__ in _ACTIVATION_THRESHOLD_RUNTIME:
        return _ACTIVATION_THRESHOLD_RUNTIME[module.__class__](module)
    return _ACTIVATION_THRESHOLDS.get(module.__class__, (0.0, _max))
