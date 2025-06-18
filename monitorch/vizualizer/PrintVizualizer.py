from typing import Any
from .AbstractVizualizer import AbstractVizualizer

class PrintVizualizer(AbstractVizualizer):

    def __init__(self):
        pass

    def vizualize(self, name : str, value : dict[str, Any]) -> None:
        for key, val in value.items():
            print(f'{name}:', key, '->', val)
