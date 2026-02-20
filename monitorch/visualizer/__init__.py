"""
Submodule implementing data visualizations.

Classes from this module encapsulate interaction with visualisation engines like Matplotlib and TensorBoard.
All interactions with vizualizers are done from :mod:`monitorch.lens` and :class:`PyTorchInspector`.
:class:`AbstractVisualizer` defines methods for vizualizers. To pass visualizer to a :class:`PyTorchInspector`,
one could pass an instance of :class:`AbstractVisualizer` or a string ``"matplotlib"``, ``"tensorboard"`` or ``"print"`` as a ``vizualizer`` argument.

Examples
--------
>>> from monitorch.inspector import PyTorchInspector
>>> from monitorch.lens import ...
>>>
>>> inspector = PyTorchInspector(
...     lenses = [...],
...     vizualizer = "tensorboard"
... )
"""

from .AbstractVisualizer import AbstractVisualizer, TagAttributes, TagType
from .MatplotlibVisualizer import MatplotlibVisualizer
from .PrintVisualizer import PrintVisualizer
from .RecorderPlayerVisualizer import PlayerVisualizer, RecorderVisualizer
from .TensorBoardVisualizer import TensorBoardVisualizer

_vizualizer_dict: dict[str, type[AbstractVisualizer]] = {'print': PrintVisualizer, 'tensorboard': TensorBoardVisualizer, 'matplotlib': MatplotlibVisualizer}

__all__ = [
    'AbstractVisualizer',
    'MatplotlibVisualizer',
    'PlayerVisualizer',
    'PrintVisualizer',
    'RecorderVisualizer',
    'TagAttributes',
    'TagType',
    'TensorBoardVisualizer',
]
