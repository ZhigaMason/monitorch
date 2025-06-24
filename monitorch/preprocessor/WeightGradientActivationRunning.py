from typing import Any

from torch import ones_like, minimum

from .AbstractBackwardPreprocessor import AbstractBackwardPreprocessor
from ._module_classes import islinear

from monitorch.numerical import RunningMeanVar

class WeightGradientActivationMemory(AbstractBackwardPreprocessor):

    def __init__(self, death : bool, eps : float = 1e-7):
        self._death = death
        self._value = {}
        self._eps = eps

    def process(self, name : str, module, grad_input, grad_output):
        grad = module.weight.grad
        if name not in self._value:
            if self._death:
                self._value[name] = (RunningMeanVar(), ones_like(grad))
            else:
                self._value[name] = RunningMeanVar()

        new_activation_tensor = grad.abs() < self._eps
        new_activation_rate = new_activation_tensor.float().mean()

        if self._death:
            activation, death_tensor = self._value[name]
            activation.update(new_activation_rate)
            death_tensor = minimum(death_tensor, (~new_activation_tensor).float())
        else:
            self._value[name].update(new_activation_rate)


    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}

    def is_preprocessing(self, module) -> bool:
        return islinear(module)
