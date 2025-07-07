from collections import OrderedDict
from typing import Iterable
from .abstract_lens import AbstractLens
from torch.nn import Module

from monitorch.gatherer import EpochModuleGatherer
from monitorch.preprocessor import AbstractPreprocessor, ParameterNorm as ParameterNormPreprocessor
from monitorch.vizualizer import AbstractVizualizer, TagAttributes, TagType
from monitorch.numerical import extract_point


class ParameterNorm(AbstractLens):

    def __init__(
        self,
        parameters : Iterable[str]|None = None,
        log_scale : bool = True,
        normalize_by_size : bool = False,
        inplace : bool = True,

        comparison_plot : bool = True,
        aggregation_method : str = 'mean'
    ):
        self._parameters = list(parameters) if parameters is not None else ['weight', 'bias']
        self._log_scale = log_scale
        self._preprocessor = ParameterNormPreprocessor(
            self._parameters, normalize=normalize_by_size, inplace=inplace
        )
        self._gatherers : list[EpochModuleGatherer] = []
        self._data : OrderedDict[str, OrderedDict[str, dict[str, float]]]= OrderedDict([
            (parameter_name, OrderedDict()) for parameter_name in self._parameters
        ])
        self._aggregation_method = aggregation_method

        self._comparison_plot = comparison_plot
        if self._comparison_plot:
            self._comparison_data : OrderedDict[str, OrderedDict[str, float]] = OrderedDict([
                (parameter_name, OrderedDict()) for parameter_name in self._parameters
            ])

    def register_module(
            self,
            module : Module,
            module_name : str
    ):
        if not all(hasattr(module, parameter_name) for parameter_name in self._parameters):
            return
        gatherer = EpochModuleGatherer(
            module, [self._preprocessor], module_name
        )
        self._gatherers.append(gatherer)
        for parameter_name in self._parameters:
            self._data[parameter_name][module_name] = {}

    def detach_from_module(self):
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ does not collect extern data """
        pass

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        for parameter_name in self._parameters:
            vizualizer.register_tags(
                f'{parameter_name} Norm'.title(),
                TagAttributes(
                    logy=self._log_scale,
                    big_plot=False,
                    annotate=False,
                    type=TagType.NUMERICAL
                )
            )

        if self._comparison_plot:
            for parameter_name in self._parameters:
                vizualizer.register_tags(
                    f'{parameter_name}{" Log" if self._log_scale else ""} Norm Comparison'.title(),
                    TagAttributes(
                        logy=False,
                        big_plot=True,
                        annotate=False,
                        type=TagType.RELATIONS
                    )
                )

    def finalize_epoch(self):
        for gatherer in self._gatherers:
            gatherer()

        for parameter_name in self._parameters:
            comparison_dict : OrderedDict[str, float]
            if self._comparison_plot:
                comparison_dict = self._comparison_data[parameter_name]
            tag_data_dict = self._data[parameter_name]
            total_sum = 0
            for module_name, module_data in self._preprocessor.value.items():
                pt_val = extract_point(module_data[parameter_name], self._aggregation_method)
                tag_data_dict.setdefault(module_name, {})[self._aggregation_method] = pt_val
                total_sum += pt_val
                if self._comparison_plot:
                    comparison_dict[module_name] = pt_val

            if self._comparison_plot:
                for module_name in comparison_dict:
                    comparison_dict[module_name] /= total_sum

    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        for parameter_name in self._parameters:
            vizualizer.plot_numerical_values(
                epoch, f'{parameter_name} Norm'.title(),
                self._data[parameter_name], None
            )

        if self._comparison_plot:
            for parameter_name in self._parameters:
                tag_name = f'{parameter_name}{" Log" if self._log_scale else ""} Norm Comparison'.title()
                vizualizer.plot_relations(
                    epoch, tag_name,
                    OrderedDict([ ( tag_name, self._comparison_data[parameter_name]) ])
                )

    def reset_epoch(self):
        self._preprocessor.reset()
