from collections import OrderedDict
from copy import deepcopy
from typing import Any

from torch import Tensor, no_grad

from monitorch.numerical import GeometryComputation
from monitorch.preprocessor.abstract.abstract_tensor_preprocessor import AbstractTensorPreprocessor


class ParameterDifferenceGeometry(AbstractTensorPreprocessor):
    """
    Preprocessor to keep track of parameters evolution with respect to preprocessor calls by inspecting it updates.

    Main usage is to inspect optimizer update step behaviour.

    Computes (normalized) L2 norm of parameter updates.
    Optionally computes correlation between consecutive parameter differences for further investigation,
    normalized to fit into [-1, 1] range.

    Parameters
    ----------
    correlation : bool
        Indicator if correlation must be computed.
    normalize : bool
        Indicator if gradient norm should be divided by square root of number of elements.
    inplace : bool
        Flag indicating whether to collect data inplace using :class:`RunningMeanVar` or to stack them into a list.
    """

    def __init__(self, correlation: bool, normalize: bool, inplace: bool, eps: float = 1e-8):
        self._gc_kwargs: dict[str, bool] = dict(
            normalize=normalize,
            correlation=correlation,
            inplace=inplace,
        )
        self._eps = eps
        self._value: OrderedDict[str, GeometryComputation] = OrderedDict()
        self._prev_param: dict[str, Tensor] = {}

    @no_grad
    def process_tensor(self, name: str, param: Tensor) -> None:
        """
        Computes (normalized) L2 norm and optionally correlation with previous difference.

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
        """See base class."""
        return {k: gc.value for k, gc in self._value.items()}

    def start_sync(self, dst_rank: int = 0) -> None:
        """
        Starts syncronizing the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        for gc in self._value.values():
            gc.start_sync(dst_rank=dst_rank)

    def finish_sync(self) -> None:
        """
        Finishes syncronizing the data with the dst_rank.
        """
        for gc in self._value.values():
            gc.finish_sync()

    def reset(self) -> None:
        """See base class."""
        self._value = OrderedDict()
