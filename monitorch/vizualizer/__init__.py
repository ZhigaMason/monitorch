from .AbstractVizualizer import AbstractVizualizer, TagAttributes
from .PrintVizualizer import PrintVizualizer
from .TensorBoardVizualizer import TensorBoardVizualizer

_vizualizer_dict = {
    'print' : PrintVizualizer,
    'tensorboard' : TensorBoardVizualizer
}
