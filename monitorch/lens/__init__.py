"""
Classes to interact with neural networks and visualization.

User interface encapsulating interactions
between :mod:`monitorch.gatherer`, :mod:`monitorch.preprocessor` and :mod:`monitorch.visualizer`.
Each lens is roughly associated with one of the preprocessors and allows visualizer configuration
through parameters in constructor.

Every lens supports both in-memory and in-place computation of statistics (default is in-place).
in-place computations consume roughly linear space in number of layers processed by a lens.
in-memory computations consume rougly linear space in number of layers processed by a lens times batch iterations.
Exceptions are gradient geometry lenses :class:`ParameterGradientGeometry` and :class:`OutputGradientGeometry`,
see the classes for details.

Lenses are managed by :class:`monitorch.inspector.PyTorchInspector`, whom they are passed to during initialization.

Examples
--------
>>> from monitorch.inspector import PyTorchInspector
>>> from monitorch.lens import LossMetrics, ParameterNorm
>>> 
>>> mynet = MyNeuralNetwork()
>>> 
>>> inspector = PyTorchInspector(
...     lenses = [
...         LossMetrics(),
...         ParameterNorm( normalize_by_size=True )
...     ],
...     module = mynet,
...     visualizer='tensorboard'
... )
>>> 
>>> for epoch in range(N_EPOCHS):
...     ... # training-validation loop remains unchanged
...     inspector.tick_epoch()
"""
from .abstract_lens                 import AbstractLens
from .loss_metrics                  import LossMetrics
from .output_norm                   import OutputNorm
from .output_activation             import OutputActivation
from .parameter_norm                import ParameterNorm
from .parameter_gradient_geometry   import ParameterGradientGeometry
from .parameter_gradient_activation import ParameterGradientActivation
from .output_gradient_geometry      import OutputGradientGeometry

__all__ = [
    "AbstractLens",
    "LossMetrics",
    "OutputNorm",
    "OutputActivation",
    "ParameterNorm",
    "ParameterGradientGeometry",
    "ParameterGradientActivation",
    "OutputGradientGeometry",
]
