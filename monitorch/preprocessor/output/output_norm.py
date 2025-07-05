
from torch import no_grad
from torch.linalg import vector_norm
from math import sqrt

from typing import Any
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor
from monitorch.numerical import RunningMeanVar

class OutputNorm(AbstractForwardPreprocessor):

    def __init__(self, normalize : bool, inplace : bool):
        self._normalize = normalize
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list

    @no_grad
    def process_fw(self, name : str, module, layer_input, layer_output):
        norm = self._value.setdefault(name, self._agg_class())
        if self._normalize:
            norm.append(vector_norm(layer_output).item() / sqrt(layer_output.numel()))
        else:
            norm.append(vector_norm(layer_output).item())

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
