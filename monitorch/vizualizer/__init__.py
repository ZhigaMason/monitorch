from .AbstractVizualizer import AbstractVizualizer
from .PrintVizualizer import PrintVizualizer
from .TensorBoardVizualizer import TensorBoardVizualizer

_vizualizer_dict = {
    'print' : PrintVizualizer,
    'tensorboard' : TensorBoardVizualizer
}
