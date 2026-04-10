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
from .PrintVisualizer import PrintVisualizer
from .RecorderPlayerVisualizer import PlayerVisualizer, RecorderVisualizer

_vizualizer_dict: dict[str, type[AbstractVisualizer]] = {'print': PrintVisualizer}

try:
    from .MatplotlibVisualizer import MatplotlibVisualizer
    _vizualizer_dict['matplotlib'] = MatplotlibVisualizer
except ImportError:
    pass

try:
    from .TensorBoardVisualizer import TensorBoardVisualizer
    _vizualizer_dict['tensorboard'] = TensorBoardVisualizer
except ImportError:
    pass

_OPTIONAL_DEPS = {
    'MatplotlibVisualizer': 'matplotlib',
    'TensorBoardVisualizer': 'tensorboard',
}


def __getattr__(name: str):
    if name in _OPTIONAL_DEPS:
        dep = _OPTIONAL_DEPS[name]
        raise ImportError(
            f"{name} requires '{dep}' to be installed. "
            f"Install it with: pip install 'monitorch[{dep}]'"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
