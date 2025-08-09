"""
A submodule implementing functions to get class of abstract ``torch.nn.Module``.

Examples
--------
>>> import torch.nn as nn
>>> from monitorch.lens.module_distinction import isactivation, isconv
>>> isactivation(nn.ReLU())
True
>>> isactivation(nn.Dropout())
False
>>> isconv(nn.BatchNorm1d(10))
False
>>> isconv(nn.Conv2d(1, 1, 1))
True
"""
from torch.nn.modules import (
    Module,

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
    #     MultiheadAttention,
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

    Bilinear,
    Identity,
    LazyLinear,
    Linear,

    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    SyncBatchNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d,

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

    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
)

_DROPOUT = {
    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
}

_ACTIVATION = {
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
    #     MultiheadAttention,
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
}

_LINEAR = {
    Bilinear,
    Identity,
    LazyLinear,
    Linear,
}

_CONV = {
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
}

_NORMALIZATION = {
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    SyncBatchNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d,
}

def isactivation(module : Module) -> bool:
    """
    Checks if provided module is an activation function module.
    Returns ``False`` for ``torch.nn.MultiheadAttention``.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be checked
    """
    return module.__class__ in _ACTIVATION

def isdropout(module : Module) -> bool:
    """
    Checks if provided module is a dropout module.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be checked
    """
    return module.__class__ in _DROPOUT

def islinear(module : Module) -> bool:
    """
    Checks if provided module is a linear non-convolution module.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be checked
    """
    return module.__class__ in _LINEAR

def isconv(module : Module) -> bool:
    """
    Checks if provided module is a convolution module.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be checked
    """
    return module.__class__ in _CONV

def isnormalization(module : Module) -> bool:
    """
    Checks if provided module is a normalization module.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be checked
    """
    return module.__class__ in _NORMALIZATION
