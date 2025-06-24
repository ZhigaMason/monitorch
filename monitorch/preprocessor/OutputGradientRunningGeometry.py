
from math import sqrt
from typing import Any
from torch import no_grad
from torch.nn import Module
from torch.linalg import vector_norm

from .AbstractBackwardPreprocessor import AbstractBackwardPreprocessor
from monitorch.numerical import RunningMeanVar
from ._module_classes import islinear

class OutputGradientRunningGeometry(AbstractBackwardPreprocessor):

    def __init__(self, adj_prod : bool, normalize : bool):
        self._adj_prod = adj_prod
        self._normalize = normalize
        self._value = {} # Either name : norm or name : (norm, prod)
        if adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}

    @no_grad
    def process(self, name : str, module, grad_input, grad_output) -> None:
        grad = grad_output
        new_norm = vector_norm(grad)
        if self._adj_prod:

            # Computes dot product of normalised current and previous gradients
            new_prod = (grad * self._prev_grad.get(name, 0.0)).sum() / (new_norm * self._prev_norm.get(name, 1.0))

            self._prev_grad[name] = grad
            self._prev_norm[name] = new_norm

            norm, prod = self._value.setdefault(name, (RunningMeanVar(), RunningMeanVar()))
            norm.update( (new_norm / sqrt(grad.numel())) if self._normalize else new_norm)
            prod.update(new_prod)

        else:
            norm = self._value.setdefault(name, RunningMeanVar())
            norm.update( (new_norm / sqrt(grad.numel())) if self._normalize else new_norm)

    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}

    def is_preprocessing(self, module : Module) -> bool:
        """ Determines whether given module is observed with the preprocessor """
        return islinear(module)
