
from typing import Any
from torch import no_grad
from torch.nn import Module
from torch.linalg import vector_norm

from .AbstractBackwardPreprocessor import AbstractBackwardPreprocessor
from monitorch.numerical import RunningMeanVar
from ._module_classes import islinear

class GradientRunningGeometry(AbstractBackwardPreprocessor):

    def __init__(self, adj_prod : bool):
        self._adj_prod = adj_prod
        self._value = {} # Either name : norm or name : (norm, prod)
        if adj_prod:
            self._prev_grad = 0.0
            self._prev_norm = 1.0

    @no_grad
    def process(self, name : str, _, grad_output) -> None:
        if self._adj_prod:
            new_norm = vector_norm(grad_output)

            # Computes dot product of normalised current and previous gradients
            new_prod = (grad_output * self._prev_grad).sum() / (new_norm * self._prev_norm)

            self._prev_grad = grad_output
            self._prev_norm = new_norm

            norm, prod = self._value.setdefault(name, (RunningMeanVar(), RunningMeanVar()))
            norm.update(new_norm)
            prod.update(new_prod)

        else:
            norm = self._value.setdefault(name, RunningMeanVar())
            norm.update(vector_norm(grad_output))

    def value(self) -> dict[str, Any]:
        """ Value computed by Abstract Preprocessor for all layers, that it is processing, identified by name"""
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = None

    def is_preprocessing(self, module : Module) -> bool:
        """ Determines whether given module is observed with the preprocessor """
        return islinear(module)
