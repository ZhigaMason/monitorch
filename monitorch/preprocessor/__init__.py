"""
Submodule for aggregating and preprocessing raw data from neural net layers.

Classes from this module implement computations of different variables that are collected during training.
Those include but are not limited to activations of neurons and norms of vectorized tensors appearing in neural net.
Those values can come from output, parameters or gradients of layers.
It is expected that, preprocessors are called from one of objects in :mod:`monitorch.gatherer`.
"""

from .abstract import AbstractBackwardPreprocessor, AbstractForwardPreprocessor, AbstractModulePreprocessor, AbstractPreprocessor, AbstractTensorPreprocessor
from .ExplicitCall import ExplicitCall
from .gradient import GradientActivation, GradientGeometry, OutputGradientGeometry
from .output import LossModule, OutputActivation, OutputNorm
from .parameter import ParameterDifferenceGeometry, ParameterNorm

__all__ = [
    'AbstractBackwardPreprocessor',
    'AbstractForwardPreprocessor',
    'AbstractModulePreprocessor',
    'AbstractPreprocessor',
    'AbstractTensorPreprocessor',
    'ExplicitCall',
    'GradientActivation',
    'GradientGeometry',
    'LossModule',
    'OutputActivation',
    'OutputGradientGeometry',
    'OutputNorm',
    'ParameterDifferenceGeometry',
    'ParameterNorm',
]
