
from collections import OrderedDict
from math import sqrt
from copy import deepcopy
from typing import Any
from torch.linalg import vector_norm

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractTensorPreprocessor
from monitorch.numerical import RunningMeanVar

class GradientGeometry(AbstractTensorPreprocessor):
    """
    Preprocessor to keep track of parameters' gradients.

    Computes (normalized) L2 norm of gradient tensor.
    Optionally computes vectorized scalar product between consecutive gradients for further gradient oscilations investigation,
    normalized to fit into [-1, 1] range.

    Parameters
    ----------
    adj_prod : bool
        Indicator if adjacent scalar product must be computed.
    normalize : bool
        Indicator if gradient norm should be divided by square root of number of elements.
    inplace : bool
        Flag indicating whether to collect data inplace using :class:`RunningMeanVar` or to stack them into a list.
    """

    def __init__(self, adj_prod : bool, normalize : bool, inplace : bool):
        self._adj_prod = adj_prod
        self._normalize = normalize
        self._value = OrderedDict() # Either name : norm or name : (norm, prod)
        self._agg_class = RunningMeanVar if inplace else list
        if adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}

    def process_tensor(self, name : str, grad) -> None:
        """
        Computes (normalized) L2 norm and optionally scalar product with previous gradient.

        The first gradient is taken to be 0.0 with norm 1.0.

        Parameters
        ----------
        name : str
            Name of source of gradient.
        grad : torch.Tensor
            Gradient tensor to be processed.
        """
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
        """ See base class. """
        return self._value

    def reset(self) -> None:
        """ See base class. """
        self._value = {}
        if self._adj_prod:
            self._prev_grad = {}
            self._prev_norm = {}
