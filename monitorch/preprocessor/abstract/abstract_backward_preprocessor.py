"""
    Base class for all backward pass preprocessors
"""

from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractBackwardPreprocessor(AbstractPreprocessor):
    """
    Base class for all preprocessors that aggregate data obtain from backward pass.

    Subclasses of ``AbstractBackwardPreprocessor`` process gradients with respect to inputs or outputs of module.
    """

    @abstractmethod
    def process_bw(self, name : str, module, grad_input, grad_output):
        """
        Processes backward pass data.

        Parameters
        ----------
        name : str
            Name of the module, its data is processed
        module : torch.nn.Module
            Module object from which the data is processed
        grad_input : torch.Tensor
            Gradients with respect to input of module.
        grad_output : torch.Tensor
            Gradients with respect to output of module.
        """
        pass
