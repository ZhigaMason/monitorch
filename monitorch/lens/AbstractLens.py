from abc import ABC, abstractmethod
from typing import Any, Type
from monitorch.preprocessor import AbstractForwardPreprocessor, AbstractBackwardPreprocessor, AbstractPreprocessor
from monitorch.vizualizer import AbstractVizualizer


class AbstractLens(ABC):

    @abstractmethod
    def requires_forward(self) -> set[Type[AbstractForwardPreprocessor]]:
        pass

    @abstractmethod
    def requires_backward(self) -> set[Type[AbstractBackwardPreprocessor]]:
        pass

    @abstractmethod
    def register_preprocessors(self, preprocessors : dict[Type[AbstractPreprocessor], AbstractPreprocessor]):
        pass

    @abstractmethod
    def vizualize(self, vizualizer : AbstractVizualizer):
        pass
