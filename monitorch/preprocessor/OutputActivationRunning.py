
from monitorch.numerical import RunningMeanVar

from torch import no_grad, ones_like, bool as bool_
from typing import Any
from .AbstractForwardPreprocessor import AbstractForwardPreprocessor
from ._module_classes import isactivation, threshold_for_module


class OutputActivationRunning(AbstractForwardPreprocessor):

    def __init__(self, death : bool, eps : float = 1e-7):
        self._death = death
        self._value = {} # Either name : activation or name : (activation, death_tensor)
        self._thresholds : dict[str, tuple[float, float]]= {}
        self._eps : float = eps

    @no_grad
    def process(self, name : str, module, layer_input, layer_output) -> None:
        if name not in self._value:
            if self._death:
                self._value[name] = (RunningMeanVar(), ones_like(layer_output, dtype=bool_))
            else:
                self._value[name] = RunningMeanVar()
            self._thresholds[name] = threshold_for_module(module)

        lo, up = self._thresholds[name]

        new_activation_tensor = ((layer_output - lo).abs() > self._eps) & ((layer_output - up).abs() > self._eps)
        new_activation_rate = new_activation_tensor.float().mean()

        if self._death:
            activation, death_tensor = self._value[name]
            activation.update(new_activation_rate)
            death_tensor &= ~new_activation_tensor
        else:
            self._value[name].update(new_activation_rate)


    @property
    def value(self) -> dict[str, Any]:
        """ Value computed by Abstract Preprocessor for all layers, that it is processing, identified by name"""
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}

    def is_preprocessing(self, module) -> bool:
        return isactivation(module)

