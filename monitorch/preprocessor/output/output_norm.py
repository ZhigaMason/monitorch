
from torch import no_grad, is_grad_enabled
from torch.linalg import vector_norm
from math import sqrt

from typing import Any
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor
from monitorch.numerical import RunningMeanVar

class OutputNorm(AbstractForwardPreprocessor):

    def __init__(self, normalize : bool, inplace : bool, record_no_grad : bool):
        self._normalize = normalize
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._agg_class = RunningMeanVar if inplace else list
        self._record_no_grad = record_no_grad

    def process_fw(self, name : str, module, layer_input, layer_output):
        if not (self._record_no_grad or is_grad_enabled()):
            return
        norm_container = self._value.setdefault(name, self._agg_class())

        with no_grad():
            norm_mean = vector_norm(layer_output.flatten(1, -1), dim=-1).mean().item()
            if self._normalize:
                norm_mean /= sqrt(layer_output[0].numel())
            norm_container.append(norm_mean)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
