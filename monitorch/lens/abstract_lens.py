from abc import ABC, abstractmethod
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor
from monitorch.vizualizer import AbstractVizualizer


class AbstractLens(ABC):

    @abstractmethod
    def register_module(self, module : Module, module_name : str):
        pass

    @abstractmethod
    def detach_from_module(self):
        pass

    @abstractmethod
    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ gets a reference of foreign preprocessor to collect data from """
        pass

    @abstractmethod
    def introduce_tags(self, vizualizer : AbstractVizualizer):
        pass

    @abstractmethod
    def finalize_epoch(self):
        pass

    @abstractmethod
    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        pass
