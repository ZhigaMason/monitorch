"""
    Base class for all forward pass preprocessors
"""

from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractForwardPreprocessor(AbstractPreprocessor):
    """
    Base class for all preprocessors that aggregate data obtain from forward pass.

    Subclasses' of ``AbstractForwardPreprocessor`` process input and output of module.
    Expects module to take a single tensor and output a single tensor, hence feed-forward preprocessor.
    """

    @abstractmethod
    def process_fw(self, name : str, module, layer_input, layer_output):
        """
        Processes forward pass data.

        Parameters
        ----------
        name : str
            Name of the module which data is processed.
        module : torch.nn.Module
            The module which inputs and outputs are processed.
        layer_input : torch.Tensor
            Input to the module.
        layer_output : torch.Tensor
            Output of the module.
        """
        pass
