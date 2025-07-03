
from math import sqrt
from copy import deepcopy
from typing import Any
from torch.linalg import vector_norm

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractGradientPreprocessor
from monitorch.numerical import RunningMeanVar

class GradientGeometryRunning(AbstractGradientPreprocessor):

    def __init__(self, adj_prod : bool, normalize : bool):
        self._adj_prod = adj_prod
        self._normalize = normalize
        self._value = {} # Either name : norm or name : (norm, prod)
        if adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}

    def process_grad(self, name : str, grad) -> None:
        new_norm = vector_norm(grad).item()
        if self._normalize:
            new_norm /= sqrt(grad.numel())
        if self._adj_prod:
            new_prod = (grad * self._prev_grad.get(name, 0.0)).sum().item() / (new_norm * self._prev_norm.get(name, 1.0))
            if self._normalize:
                new_prod /= grad.numel()

            self._prev_grad[name] = deepcopy(grad)
            self._prev_norm[name] = new_norm

            norm, prod = self._value.setdefault(name, (RunningMeanVar(), RunningMeanVar()))
            norm.update(new_norm)
            prod.update(new_prod)

        else:
            norm = self._value.setdefault(name, RunningMeanVar())
            norm.update(new_norm)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}
