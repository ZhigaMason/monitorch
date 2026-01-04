
from math import sqrt
from copy import deepcopy
from typing import Any
from torch import no_grad
from torch.nn import Module
from torch.linalg import vector_norm

from monitorch.preprocessor.abstract.abstract_backward_preprocessor import AbstractBackwardPreprocessor
from monitorch.numerical import RunningMeanVar, GeometryComputation

from collections import OrderedDict

class OutputGradientGeometry(AbstractBackwardPreprocessor):
    """
    Preprocessor to keep track of outputs' gradients.

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

    def __init__(self, adj_prod : bool, normalize : bool, inplace : bool, eps : float = 1e-8):
        self._gc_kwargs : dict[str, bool] = dict(
            normalize=normalize,
            dot_product=adj_prod,
            inplace=inplace,
        )
        self._eps = eps
        self._value : OrderedDict[str, GeometryComputation]= OrderedDict()


    @no_grad
    def process_bw(self, name : str, module, grad_input, grad_output) -> None:
        """
        Computes (normalized) L2 norm and optionally computes scalar product with previous output's gradient.

        The first gradient is taken to be 0.0 with norm 1.0.

        Parameters
        ----------
        name : str
            Name of the module which output's gradients to record.
        moduel : torch.nn.Module
            The module object. Ignored.
        grad_input
            Gradients with respect to input of layer. Ignored.
        grad_output
            Gradients with respect to outputs of layer.
            Assumes layer outputs single tensor, thus having single output gradient.
        """
        grad = grad_output[0]
        geometry_computation = self._value.setdefault(name, GeometryComputation(**self._gc_kwargs, eps=self._eps))
        geometry_computation.update(grad)

    @property
    def value(self) -> dict[str, Any]:
        """ See base class. """
        return {k:gc.value for k,gc in self._value.items()}

    def reset(self) -> None:
        """ See base class. """
        self._value = OrderedDict()
