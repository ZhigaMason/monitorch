from abc import ABC, abstractmethod
from monitorch.preprocessor import ExplicitCall
from monitorch.vizualizer import AbstractVizualizer


class AbstractLens(ABC):

    @abstractmethod
    def register_module(self, module):
        pass

    @abstractmethod
    def register_explicit_call_ppr(self, ecppr : ExplicitCall):
        pass

    @abstractmethod
    def register_tags(self, vizualizer : AbstractVizualizer):
        pass

    @abstractmethod
    def vizualize(self, vizualizer : AbstractVizualizer):
        pass
