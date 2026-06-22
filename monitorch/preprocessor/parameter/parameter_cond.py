from collections import OrderedDict
from typing import Any

from torch.linalg import cond

from monitorch.numerical import RunningMeanVar
from monitorch.preprocessor.abstract.abstract_module_preprocessor import AbstractModulePreprocessor


class ParameterCond(AbstractModulePreprocessor):
    """
    Preprocessor computing conditonal number of parameters.

    Computes conditonal number of parameters listed in :attr:`attrs_`
    for every module that is being passed to process module.

    Parameters
    ----------
    attrs : list[str]
        List of attributes for which norm will be computed.
    normalize : bool
        Flag indicating whether norm should be normalized by tensor size.
        If true computes RMS of tensor values, L2-norm otherwise.
    inplace : bool
        Flag indicating if :class:`RunningMeanVar` or ``list`` will be used.

    Attributes
    ----------
    attrs_ : list[str]
        List of attributes to compute norm for.
    """

    def __init__(self, attrs: list[str], inplace: bool):
        self.attrs_ = attrs
        self.accumulator = RunningMeanVar if inplace else list
        self._value: OrderedDict[str, dict[str, self.accumulator]] = OrderedDict()

    def process_module(self, name: str, module):
        """
        Computes norms of all :attr:`attrs_`.

        Uses ``torch.linalg.vector_norm`` to compute L2-norm of module's attributes.
        If ``normalize`` is true, divides norm by a square root of number of elements in attributes.
        """
        d = self._value.setdefault(name, {})
        for attr in self.attrs_:
            param = getattr(module, attr)
            val = d.setdefault(attr, self.accumulator())
            val.update(cond(param, p=-2))

    @property
    def value(self) -> OrderedDict[str, Any]:
        """
        See base class
        """
        return OrderedDict([(name, {attr: d[attr] for attr in self.attrs_}) for name, d in self._value.items()])

    def start_sync(self, dst_rank: int = 0) -> None:
        """
        Starts synchronization the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        for val in self._value.values():
            for rmv in val.values():
                rmv.start_sync(dst_rank=dst_rank)

    def finish_sync(self) -> None:
        """
        Starts synchronization the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        for val in self._value.values():
            for rmv in val.values():
                rmv.finish_sync()

    def reset(self) -> None:
        """
        See base class
        """
        self._value = OrderedDict()
