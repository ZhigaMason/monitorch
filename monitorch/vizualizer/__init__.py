"""
Submodule implementing data visualizations.

Classes from this module encapsulate interaction with visualisation engines like Matplotlib and TensorBoard.
All interactions with vizualizers are done from :mod:`monitorch.lens` and :class:`PyTorchInspector`.
:class:`AbstractVizualizer` defines methods for vizualizers. To pass visualizer to a :class:`PyTorchInspector`,
one could pass an instance of :class:`AbstractVizualizer` or a string ``"matplotlib"``, ``"tensorboard"`` or ``"print"`` as a ``vizualizer`` argument.

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
from .AbstractVizualizer import AbstractVizualizer, TagAttributes, TagType
from .PrintVizualizer import PrintVizualizer
from .TensorBoardVizualizer import TensorBoardVizualizer
from .MatplotlibVizualizer import MatplotlibVizualizer

_vizualizer_dict : dict[str, Type[AbstractVizualizer]] = {
    'print'       : PrintVizualizer,
    'tensorboard' : TensorBoardVizualizer,
    'matplotlib'  : MatplotlibVizualizer
}

__all__ = [
    "PrintVizualizer",
    "TensorBoardVizualizer",
    "MatplotlibVizualizer"
]
