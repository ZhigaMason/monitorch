from typing import Iterable
from collections import OrderedDict
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor, GradientGeometry
from monitorch.vizualizer import AbstractVizualizer
from monitorch.gatherer import ParameterGradientGatherer


class ParameterGradientGeometry(AbstractLens):

    def __init__(
        self,
        inplace : bool = True,
        normalize_by_size : bool = False,
        log_scale : bool = False,

        compute_adj_prod : bool = True,

        parameters : str|Iterable[str] = ('weight', 'bias'),

        line_aggregation : str|Iterable[str] = 'mean',
        range_aggregation : str|Iterable[str]|None = ('std', 'min-max')
    ):
        self._parameters = parameters
        self._compute_adj_prod = compute_adj_prod
        self._preprocessors = OrderedDict([
            (parameter, GradientGeometry(inplace=inplace, normalize=normalize_by_size, adj_prod=compute_adj_prod))
            for parameter in self._parameters
        ])
        self._gatherers = []
        self._data : dict[str, OrderedDict[str, float]]= {}
        if self._compute_adj_prod:
            self._adj_prod_data : dict[str, OrderedDict[str, float]]= {}
        self._log_scale = log_scale


        self._line_aggregation : Iterable[str] = [line_aggregation] if isinstance(line_aggregation, str) else line_aggregation
        self._range_aggregation : Iterable[str]
        if isinstance(range_aggregation, str):
            self._range_aggregation = [range_aggregation]
        elif range_aggregation is None:
            self._range_aggregation = []
        else:
            self._range_aggregation = range_aggregation

    def register_module(self, module : Module, module_name : str):
        if not all(hasattr(module, parameter_name) for parameter_name in self._parameters):
            return

        for parameter in self._parameters:
        #    gg = ParameterGradientGatherer()
            pass

    def detach_from_module(self):
        pass

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ gets a reference of foreign preprocessor to collect data from """
        pass

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        pass

    def finalize_epoch(self):
        pass

    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        pass

    def reset_epoch(self):
        pass
