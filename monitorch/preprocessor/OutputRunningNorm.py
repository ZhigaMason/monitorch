
from torch import no_grad
from torch.linalg import vector_norm
from math import sqrt

from typing import Any
from .AbstractForwardPreprocessor import AbstractForwardPreprocessor
from ._module_classes import isactivation
from monitorch.numerical import RunningMeanVar

class OutputRunningNorm(AbstractForwardPreprocessor):

    def __init__(self, normalize : bool):
        self._normalize = normalize
        self._value = {}

    @no_grad
    def process(self, name : str, module, layer_input, layer_output):
        norm = self._value.setdefault(name, RunningMeanVar())
        if self._normalize:
            norm.update(vector_norm(layer_output) / sqrt(layer_output.numel()))
        else:
            norm.update(vector_norm(layer_output))

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}

    def is_preprocessing(self, module) -> bool:
        return isactivation(module)

