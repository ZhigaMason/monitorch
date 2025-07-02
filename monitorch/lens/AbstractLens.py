from abc import ABC, abstractmethod
from typing import Any, Type
from monitorch.preprocessor import AbstractForwardPreprocessor, AbstractBackwardPreprocessor, AbstractPreprocessor
from monitorch.vizualizer import AbstractVizualizer


class AbstractLens(ABC):

    @abstractmethod
    def register_module(self, module):
        pass

    @abstractmethod
    def register_tags(self, vizualizer : AbstractVizualizer):
        pass
