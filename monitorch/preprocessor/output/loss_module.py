from typing import Any
from torch import is_grad_enabled
from monitorch.preprocessor.abstract import AbstractForwardPreprocessor
from monitorch.numerical import RunningMeanVar

class LossModule(AbstractForwardPreprocessor):

    def __init__(self, inplace : bool):
        self._value = {}
        self._train_str_loss = ''
        self._non_train_str_loss = ''
        self._agg_class = RunningMeanVar if inplace else list

    def set_loss_strs(self, train_loss_str : str, non_train_loss_str : str):
        self._value = {
            train_loss_str : self._agg_class(),
            non_train_loss_str : self._agg_class()
        }
        self._train_str_loss = train_loss_str
        self._non_train_str_loss = non_train_loss_str

    def process_fw(self, name : str, module, layer_input, layer_output):
        assert layer_output.numel() == 1, "Only single item loss can be preprocessed"
        if is_grad_enabled():
            self._value[self._train_str_loss].append(layer_output.item())
        else:
            self._value[self._non_train_str_loss].append(layer_output.item())

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {
            self._train_str_loss : RunningMeanVar(),
            self._non_train_str_loss : RunningMeanVar()
        }


