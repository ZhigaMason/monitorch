
from typing import Any
from math import sqrt
from torch.linalg import vector_norm
from monitorch.preprocessor.abstract.abstract_module_preprocessor import AbstractModulePreprocessor

class ParameterNormMemory(AbstractModulePreprocessor):

    def __init__(self, attrs : list[str], normalize : bool):
        self._normalize = normalize
        self._attrs = attrs
        self._value : dict[str, dict[str, list[float]]]= {}

    def process_module(self, name : str, module):
        d = self._value.setdefault(name, {})
        for attr in self._attrs:
            norm = vector_norm(getattr(module, attr)).item()
            if self._normalize:
                norm /= sqrt(getattr(module, attr).numel())
            d.setdefault(attr, []).append(norm)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
