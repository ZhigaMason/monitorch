from typing import Any
from torch import is_grad_enabled
from monitorch.preprocessor.abstract import AbstractForwardPreprocessor
from monitorch.numerical import RunningMeanVar

class LossModule(AbstractForwardPreprocessor):
    """
    Module to record single value loss.

    Aggregates loss from loss modules (i.e. ``torch.nn.MSELoss`` or ``torch.nn.NLLLoss``).
    It can be accessed later.

    Parameters
    ----------
    inplace : bool
        Indicator if :class:`RunningMeanVar` or ``list`` should be used for aggregation.
    """

    def __init__(self, inplace : bool):
        self._value = {}
        self._train_str_loss = ''
        self._non_train_str_loss = ''
        self._agg_class = RunningMeanVar if inplace else list

    def set_loss_strs(self, train_loss_str : str, non_train_loss_str : str):
        """
        Defines names for training and test/validation/development loss.
        Given strings will be used in :meth:`value` for indexing.

        Parameters
        ----------
        train_loss_str : str
            String used for training loss.
        non_train_loss_str : str
            String used for test/validation/development loss.
        """
        self._value = {
            train_loss_str : self._agg_class(),
            non_train_loss_str : self._agg_class()
        }
        self._train_str_loss = train_loss_str
        self._non_train_str_loss = non_train_loss_str

    def process_fw(self, name : str, module, layer_input, layer_output):
        """
        Saves loss passed as layer output.

        Parameters
        ----------
        name : str
            Name of the module. Ignored.
        module : torch.nn.Module
            The module object. Ignored.
        layer_input : torch.Tensor
            Input to loss module. Ignored.
        layer_output : torch.Tensor
            Loss tensor. Must have single element.

        Raises
        ------
        AttributeError
            If layer_output has none or more than one elements.
        """
        if layer_output.numel() != 1:
            raise AttributeError("Only single item loss can be preprocessed")
        if is_grad_enabled():
            self._value[self._train_str_loss].append(layer_output.item())
        else:
            self._value[self._non_train_str_loss].append(layer_output.item())

    @property
    def value(self) -> dict[str, Any]:
        """ See base class. """
        return self._value

    def reset(self) -> None:
        """ See base class. """
        self._value = {
            self._train_str_loss : self._agg_class(),
            self._non_train_str_loss : self._agg_class()
        }
