
from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractModulePreprocessor(AbstractPreprocessor):
    """
    Base class for all preprocessors that process module on its own.

    Does not restrict usage by requiring inputs, outputs or gradients of module.
    """

    @abstractmethod
    def process_module(self, name : str, module):
        """
        Processes module.

        Parameters
        ----------
        name : str
            Name of the module.
        module : torch.nn.Module
            The module object.
        """
        pass
