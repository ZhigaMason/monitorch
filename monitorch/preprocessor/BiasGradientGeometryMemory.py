
from math import sqrt
from typing import Any
from torch.linalg import vector_norm
from torch import no_grad

from ._module_classes import islinear
from .AbstractGradientPreprocessor import AbstractGradientPreprocessor


class BiasGradientGeometryMemory(AbstractGradientPreprocessor):

    def __init__(self, adj_prod, normalize):
        self._adj_prod = adj_prod
        self._normalize = normalize
        self._value = {} # Either name : norm or name : (norm, prod)
        if adj_prod:
            self._prev_grad = {}

    def process_grad(self, name : str, grad) -> None:
        l = self._value.setdefault(name, [])
        new_norm = vector_norm(grad).item()
        if self._normalize:
            new_norm /= sqrt(grad.numel())

        if self._adj_prod:

            # Computes dot product of normalised current and previous gradients
            prev_norm = l[-1][0] if l else 1.0
            if self._normalize:
                prev_norm *= sqrt(grad.numel())
            new_prod = ( grad * self._prev_grad.get(name, 0.0)).sum().item() / ( new_norm * prev_norm)

            self._prev_grad[name] = grad
            l.append( (new_norm, new_prod) )
        else:
            l.append(new_norm)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = {}

    def is_preprocessing(self, module) -> bool:
        """ Determines whether given module is observed with the preprocessor """
        return islinear(module)
