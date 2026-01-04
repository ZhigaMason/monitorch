
from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractTensorPreprocessor(AbstractPreprocessor):
    """
    Base class for all preprocessors that process single tensor.

    Subclasses are mostly preprocessors that process gradient obtained during backward pass.
    Those preprocessors cannot be made :class:`AbstractBackwardPreprocessor`, because backward hooks
    are executed before gradients in tensors were updated.
    """

    @abstractmethod
    def process_tensor(self, name, tensor):
        """
        Processes tensor.

        Parameters
        ----------
        name : str
            Name of the source of tensor
        tensor : torch.Tensor
            Tensor to be processed
        """
        pass

