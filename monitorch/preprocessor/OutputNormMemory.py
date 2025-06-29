
from torch import no_grad
from torch.linalg import vector_norm
from math import sqrt

from typing import Any
from .AbstractForwardPreprocessor import AbstractForwardPreprocessor
from ._module_classes import isactivation

class OutputNormMemory(AbstractForwardPreprocessor):

    def __init__(self, normalize : bool):
        self._normalize = normalize
        self._value = {}

    @no_grad
    def process(self, name : str, module, layer_input, layer_output):
        norm = self._value.setdefault(name, [])
        if self._normalize:
            norm.append(vector_norm(layer_output).item() / sqrt(layer_output.numel()))
        else:
            norm.append(vector_norm(layer_output).item())

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}

    def is_preprocessing(self, module) -> bool:
        return isactivation(module)

