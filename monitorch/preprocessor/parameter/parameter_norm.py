
from collections import OrderedDict
from typing import Any
from math import sqrt
from torch.linalg import vector_norm
from monitorch.preprocessor.abstract.abstract_module_preprocessor import AbstractModulePreprocessor
from monitorch.numerical import RunningMeanVar


class ParameterNorm(AbstractModulePreprocessor):

    def __init__(self, attrs : list[str], normalize : bool, inplace : bool):
        self._normalize = normalize
        self._attrs = attrs
        self._value = OrderedDict()
        self._agg_class = RunningMeanVar if inplace else list

    def process_module(self, name : str, module):
        d = self._value.setdefault(name, {})
        for attr in self._attrs:
            norm = vector_norm(getattr(module, attr)).item()
            if self._normalize:
                norm /= sqrt(getattr(module, attr).numel())
            d.setdefault(attr, self._agg_class() ).append(norm)

    @property
    def value(self) -> OrderedDict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = OrderedDict()
