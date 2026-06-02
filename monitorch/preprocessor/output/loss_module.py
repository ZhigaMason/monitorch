from typing import Any

from monitorch.numerical import RunningMeanVar, finish_sync_rmv_or_error, start_sync_rmv_or_error
from monitorch.preprocessor.abstract import AbstractForwardPreprocessor

from .utils import make_train_switch


class LossModule(AbstractForwardPreprocessor):
    """
    Module to record single value loss.

    Aggregates loss from loss modules (i.e. ``torch.nn.MSELoss`` or ``torch.nn.NLLLoss``).
    It can be accessed later.

    Parameters
    ----------
    inplace : bool
        Indicator if :class:`RunningMeanVar` or ``list`` should be used for aggregation.
    evaluation_from_grad : bool
        Flag indicating if evaluation passes should be considered from gradient or modele.training
    """

    def __init__(self, inplace: bool, evaluation_from_grad: bool):
        self._value = {}
        self._train_str_loss = ''
        self._non_train_str_loss = ''
        self._agg_class = RunningMeanVar if inplace else list
        self._is_train = make_train_switch(evaluation_from_grad)

    def set_loss_strs(self, train_loss_str: str, non_train_loss_str: str):
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
        self._value = {train_loss_str: self._agg_class(), non_train_loss_str: self._agg_class()}
        self._train_str_loss = train_loss_str
        self._non_train_str_loss = non_train_loss_str

    def process_fw(self, name: str, module, layer_input, layer_output):
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
            raise AttributeError('Only single item loss can be preprocessed')
        if self._is_train(module):
            self._value[self._train_str_loss].append(layer_output.item())
        else:
            self._value[self._non_train_str_loss].append(layer_output.item())

    @property
    def value(self) -> dict[str, Any]:
        """See base class."""
        return self._value

    def start_sync(self, dst_rank: int = 0) -> None:
        """
        Start synchronization the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        for val in self._value.values():
            start_sync_rmv_or_error(val, dst_rank=dst_rank)

    def finish_sync(self) -> None:
        """
        Finish synchronization the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        for val in self._value.values():
            finish_sync_rmv_or_error(val)

    def reset(self) -> None:
        """See base class."""
        self._value = {self._train_str_loss: self._agg_class(), self._non_train_str_loss: self._agg_class()}
