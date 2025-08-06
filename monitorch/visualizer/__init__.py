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
from typing import Type
from .AbstractVisualizer import AbstractVisualizer, TagAttributes, TagType
from .PrintVisualizer import PrintVisualizer
from .TensorBoardVisualizer import TensorBoardVisualizer
from .MatplotlibVisualizer import MatplotlibVisualizer

_vizualizer_dict : dict[str, Type[AbstractVisualizer]] = {
    'print'       : PrintVisualizer,
    'tensorboard' : TensorBoardVisualizer,
    'matplotlib'  : MatplotlibVisualizer
}

__all__ = [
    "PrintVisualizer",
    "TensorBoardVisualizer",
    "MatplotlibVisualizer",
    "AbstractVisualizer",
    "TagAttributes",
    "TagType",
]
