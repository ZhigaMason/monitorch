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
