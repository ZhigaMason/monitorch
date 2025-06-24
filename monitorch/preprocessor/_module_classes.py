from itertools import chain

from math import pi
from torch.nn.modules import Module
from torch.nn.modules.linear import *
from torch.nn.modules.conv import *
from torch.nn.modules.activation import *

_LINEAR = [
    Bilinear,
    Identity,
    LazyLinear,
    Linear,
]

_CONVOLUTION = [
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,

]

_ACTIVATION = [
    CELU,
    ELU,
    GELU,
    GLU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    # MultiheadAttention,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    SELU,
    Sigmoid,
    SiLU,
    Softmax,
    Softmax2d,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
]

_max = float("+inf")
_min = float("-inf")

_min_max = (_min, _max)

_ACTIVATION_THRESHOLDS = {
    CELU : (-1, _max),
    ELU : (-1, _max),
    Hardsigmoid : (0, 1),
    LogSigmoid :(_min, 0),
    ReLU6 : (0, 6),
    SELU : (-1.7580993408473766, _max), # scale * alpha * (-1)
    Sigmoid : (0, 1),
    Softmax : _min_max,
    Softmax2d : _min_max,
    Softmin : _min_max,
    Softsign : (-1, 1),
    Tanh : (-pi/2, pi/2),
    Tanhshrink : _min_max,
}

_ACTIVATION_THRESHOLD_RUNTIME = {
    Threshold : lambda module: (module.value, _max),
    Hardtanh  : lambda module: (module.min_val, module.max_val)
}

def islinear(module : Module) -> bool:
    return any((
        isinstance(module, cls) for cls in chain(_LINEAR, _CONVOLUTION)
    ))

def isactivation(module : Module) -> bool:
    return any((
        isinstance(module, cls) for cls in chain(_ACTIVATION)
    ))

def threshold_for_module(module : Module|None) -> tuple[float, float]:
    if module.__class__ in _ACTIVATION_THRESHOLD_RUNTIME:
        return _ACTIVATION_THRESHOLD_RUNTIME[module.__class__](module)
    return _ACTIVATION_THRESHOLDS.get(module.__class__, (0, _max))
