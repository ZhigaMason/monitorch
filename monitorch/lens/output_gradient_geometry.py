from typing import Iterable
from collections import OrderedDict
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor, OutputGradientGeometry as OutputGradientGeometryPreprocessor
from monitorch.vizualizer import AbstractVizualizer, TagAttributes, TagType
from monitorch.gatherer import BackwardGatherer
from monitorch.numerical import extract_point, extract_range, parse_range_name

from .module_distinction import isactivation


class OutputGradientGeometry(AbstractLens):

    SMALL_NORM_TAG_NAME = "Output Gradient Norm"
    SMALL_PROD_TAG_NAME = "Output Gradient Adj Prod"

    def __init__(
        self,
        inplace : bool = True,
        normalize_by_size : bool = False,
        log_scale : bool = False,

        compute_adj_prod : bool = True,

        skip_activation : bool = True,

        line_aggregation : str|Iterable[str] = 'mean',
        range_aggregation : str|Iterable[str]|None = ('std', 'min-max')
    ):
        self._compute_adj_prod = compute_adj_prod
        self._skip_activation = skip_activation
        self._preprocessor = OutputGradientGeometryPreprocessor(inplace=inplace, normalize=normalize_by_size, adj_prod=compute_adj_prod)
        self._gatherers = []
        self._line_data : OrderedDict[str, dict[str, float]] = OrderedDict()
        self._range_data : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]] = OrderedDict()
        if self._compute_adj_prod:
            self._line_adj_prod_data : OrderedDict[str, dict[str, float]] = OrderedDict()
            self._range_adj_prod_data : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]] = OrderedDict()
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
        if self._skip_activation and isactivation(module):
            return

        bg = BackwardGatherer(
            module, [self._preprocessor], module_name
        )
        self._gatherers.append(bg)

    def detach_from_module(self):
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ no external data collection """
        pass

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        vizualizer.register_tags(
            OutputGradientGeometry.SMALL_NORM_TAG_NAME,
            TagAttributes(
                logy=self._log_scale,
                big_plot=False,
                annotate=True,
                type=TagType.NUMERICAL
            )
        )
        if self._compute_adj_prod:
            vizualizer.register_tags(
                OutputGradientGeometry.SMALL_PROD_TAG_NAME,
                TagAttributes(
                    logy=False,
                    big_plot=False,
                    annotate=True,
                    type=TagType.NUMERICAL
                )
            )

    def finalize_epoch(self):
        for module_name, value in self._preprocessor.value.items():
            line_norm_dict  : dict[str, float] = self._line_data.setdefault(module_name, {})
            range_norm_dict : dict[tuple[str, str], tuple[float, float]]= self._range_data.setdefault(module_name, {})
            line_prod_dict  : dict[str, float]
            range_prod_dict : dict[tuple[str, str], tuple[float, float]]
            if self._compute_adj_prod:
                line_prod_dict = self._line_adj_prod_data.setdefault(module_name, {})
                range_prod_dict = self._range_adj_prod_data.setdefault(module_name, {})

            if self._compute_adj_prod:
                norm, prod = value
                for method in self._line_aggregation:
                    line_norm_dict[method] = extract_point(norm, method)
                    line_prod_dict[method] = extract_point(prod, method)
                for method in self._range_aggregation:
                    range_norm_dict[parse_range_name(method)] = extract_range(norm, method)
                    range_prod_dict[parse_range_name(method)] = extract_range(prod, method)
            else:
                for method in self._line_aggregation:
                    line_norm_dict[method] = extract_point(value, method)
                for method in self._range_aggregation:
                    range_norm_dict[parse_range_name(method)] = extract_range(value, method)
        self._line_data = OrderedDict(reversed(self._line_data.items()))
        self._range_data = OrderedDict(reversed(self._range_data.items()))
        if self._compute_adj_prod:
            self._line_adj_prod_data = OrderedDict(reversed(self._line_adj_prod_data.items()))
            self._range_adj_prod_data = OrderedDict(reversed(self._range_adj_prod_data.items()))



    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        vizualizer.plot_numerical_values(
            epoch, OutputGradientGeometry.SMALL_NORM_TAG_NAME,
            self._line_data, self._range_data
        )
        if self._compute_adj_prod:
            vizualizer.plot_numerical_values(
                epoch, OutputGradientGeometry.SMALL_PROD_TAG_NAME,
                self._line_adj_prod_data, self._range_adj_prod_data
            )

    def reset_epoch(self):
        self._preprocessor.reset()
