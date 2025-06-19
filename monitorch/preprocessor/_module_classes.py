from itertools import chain

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

def islinear(module : Module) -> bool:
    return any((
        isinstance(module, cls) for cls in chain(_LINEAR, _CONVOLUTION)
    ))

def isactivation(module : Module) -> bool:
    return any((
        isinstance(module, cls) for cls in chain(_ACTIVATION)
    ))
