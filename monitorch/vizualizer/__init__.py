from .AbstractVizualizer import AbstractVizualizer, TagAttributes
from .PrintVizualizer import PrintVizualizer
from .TensorBoardVizualizer import TensorBoardVizualizer
from .MatplotlibVizualizer import MatplotlibVizualizer

_vizualizer_dict = {
    'print'       : PrintVizualizer(),
    'tensorboard' : TensorBoardVizualizer(),
    'matplotlib'  : MatplotlibVizualizer()
}
