
from collections import OrderedDict
from math import sqrt
from copy import deepcopy
from typing import Any
from torch.linalg import vector_norm
from torch import Tensor

from monitorch.preprocessor.abstract.abstract_tensor_preprocessor import AbstractTensorPreprocessor
from monitorch.numerical import GeometryComputation

class ParameterDifferenceGeometry(AbstractTensorPreprocessor):
    """
    Preprocessor to keep track of parameters evolution with respect to preprocessor calls.

    Main usage is to inspect optimizer update step behaviour.

    Computes (normalized) L2 norm of parameter tensor.
    Optionally computes vectorized scalar product between consecutive gradients for further investigation,
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

    def __init__(self, adj_prod : bool, normalize : bool, inplace : bool, eps : float = 1e-8):
        self._gc_kwargs : dict[str, bool] = dict(
            normalize=normalize,
            dot_product=adj_prod,
            inplace=inplace,
        )
        self._eps = eps
        self._value : OrderedDict[str, GeometryComputation] = OrderedDict()
        self._prev_param : dict[str, Tensor] = {}

    def process_tensor(self, name : str, param : Tensor) -> None:
        """
        Computes (normalized) L2 norm and optionally scalar product with previous difference.

        Parameters
        ----------
        name : str
            Name of source of parameter.
        param : torch.Tensor
            Parameter tensor to be processed.
        """
        if name in self._prev_param:
            diff = param - self._prev_param[name]
            geometry_computation = self._value.setdefault(name, GeometryComputation(**self._gc_kwargs, eps=self._eps))
            geometry_computation.update(diff)
        self._prev_param[name] = deepcopy(param)

    @property
    def value(self) -> dict[str, Any]:
        """ See base class. """
        return {k:gc.value for k,gc in self._value.items()}

    def reset(self) -> None:
        """ See base class. """
        self._value = OrderedDict()
