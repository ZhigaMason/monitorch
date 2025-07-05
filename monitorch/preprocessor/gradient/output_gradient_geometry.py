
from math import sqrt
from copy import deepcopy
from typing import Any
from torch import no_grad
from torch.nn import Module
from torch.linalg import vector_norm

from monitorch.preprocessor.abstract.abstract_backward_preprocessor import AbstractBackwardPreprocessor
from monitorch.numerical import RunningMeanVar

class OutputGradientGeometry(AbstractBackwardPreprocessor):

    def __init__(self, adj_prod : bool, normalize : bool, inplace : bool):
        self._adj_prod = adj_prod
        self._normalize = normalize
        self._value = {} # Either name : norm or name : (norm, prod)
        self._agg_class = RunningMeanVar if inplace else list
        if adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}

    @no_grad
    def process_bw(self, name : str, module, grad_input, grad_output) -> None:
        grad = grad_output[0]
        new_norm = vector_norm(grad).item()
        if self._normalize:
            new_norm /= sqrt(grad.numel())

        if self._adj_prod:
            new_prod = (grad * self._prev_grad.get(name, 0.0)).sum().item() / (new_norm * self._prev_norm.get(name, 1.0))
            if self._normalize:
                new_prod /= grad.numel()

            self._prev_grad[name] = deepcopy(grad)
            self._prev_norm[name] = new_norm

            norm, prod = self._value.setdefault(name, (self._agg_class(), self._agg_class()))
            norm.append(new_norm)
            prod.append(new_prod)

        else:
            norm = self._value.setdefault(name, self._agg_class())
            norm.append(new_norm)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}
