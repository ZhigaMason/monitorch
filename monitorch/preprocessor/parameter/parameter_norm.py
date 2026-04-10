from collections import OrderedDict
from typing import Any

from monitorch.numerical import GeometryComputation
from monitorch.preprocessor.abstract.abstract_module_preprocessor import AbstractModulePreprocessor


class ParameterNorm(AbstractModulePreprocessor):
    """
    Preprocessor computing norms of parameters.

    Computes norm of parameters listed in :attr:`attrs_`
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

    def __init__(self, attrs: list[str], normalize: bool, inplace: bool):
        self._gc_kwargs = dict(normalize=normalize, inplace=inplace, correlation=False, eps=0.0)
        self.attrs_ = attrs
        self._value: OrderedDict[str, dict[str, GeometryComputation]] = OrderedDict()

    def process_module(self, name: str, module):
        """
        Computes norms of all :attr:`attrs_`.

        Uses ``torch.linalg.vector_norm`` to compute L2-norm of module's attributes.
        If ``normalize`` is true, divides norm by a square root of number of elements in attributes.
        """
        d = self._value.setdefault(name, {})
        for attr in self.attrs_:
            param = getattr(module, attr)
            gc = d.setdefault(attr, GeometryComputation(**self._gc_kwargs))
            gc.update(param)

    @property
    def value(self) -> OrderedDict[str, Any]:
        """
        See base class
        """
        return OrderedDict([(name, {attr: d[attr].value for attr in self.attrs_}) for name, d in self._value.items()])

    def reset(self) -> None:
        """
        See base class
        """
        self._value = OrderedDict()
