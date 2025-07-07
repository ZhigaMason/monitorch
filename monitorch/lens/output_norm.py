from torch.nn import Module
from collections import OrderedDict
from typing import Iterable

from monitorch.preprocessor import AbstractPreprocessor, OutputNorm as OutputNormPreprocessor
from monitorch.vizualizer import AbstractVizualizer, TagAttributes, TagType
from monitorch.gatherer import FeedForwardGatherer
from monitorch.numerical import extract_point, extract_range, parse_range_name



from .module_distinction import isactivation
from .abstract_lens import AbstractLens

class OutputNorm(AbstractLens):

    SMALL_TAG_NAME = "Output Norm"

    def __init__(
            self,
            normalize_by_size : bool = True,
            log_scale : bool = True,
            inplace : bool = True,
            skip_no_grad_pass : bool = True,

            activation_only : bool = True,

            comparison_plot : bool = True,
            comparison_aggregation : str|None = None,

            line_aggregation : str|Iterable[str] = 'mean',
            range_aggregation : str|Iterable[str]|None = ('std', 'min-max')
    ):
        self._preprocessor = OutputNormPreprocessor(
            normalize=normalize_by_size,
            inplace=inplace,
            record_no_grad=not skip_no_grad_pass
        )

        self._small_tag_attr = TagAttributes(
            logy=log_scale,
            big_plot=False,
            annotate=True,
            type=TagType.NUMERICAL
        )

        self._gatherers : list[FeedForwardGatherer] = []
        self._line_data  : OrderedDict[str, dict[str, float]] = OrderedDict()
        self._range_data : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]] = OrderedDict()

        self._activation_only = activation_only

        self._line_aggregation : Iterable[str] = [line_aggregation] if isinstance(line_aggregation, str) else line_aggregation
        self._range_aggregation : Iterable[str]
        if isinstance(range_aggregation, str):
            self._range_aggregation = [range_aggregation]
        elif range_aggregation is None:
            self._range_aggregation = []
        else:
            self._range_aggregation = range_aggregation

        self._comparison_plot = comparison_plot
        if self._comparison_plot:
            self._comparison_aggregation = comparison_aggregation if comparison_aggregation else next(iter(self._line_aggregation))
            self._comparison_plot_name = f'{self._comparison_aggregation} Output{" Log" if log_scale else ""} Norm Comparison'.title()
            self._comparison_data : OrderedDict[str, float]= OrderedDict()



    def register_module(self, module : Module, module_name : str):
        if not self._activation_only or isactivation(module):
            ffg = FeedForwardGatherer(
                module, [self._preprocessor], module_name
            )
            self._gatherers.append(ffg)
            self._line_data[module_name]  = {}
            self._range_data[module_name] = {}

    def detach_from_module(self):
        for ffg in self._gatherers:
            ffg.detach()
        self._gatherers = []

    def register_foreign_preprocessor(self, _ : AbstractPreprocessor):
        """ does not interact with explicit call ppr """
        pass

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        vizualizer.register_tags(
            OutputNorm.SMALL_TAG_NAME, self._small_tag_attr
        )
        if self._comparison_plot:
            vizualizer.register_tags(
                self._comparison_plot_name,
                TagAttributes(
                    logy=False, big_plot=True, annotate=False, type=TagType.RELATIONS
                )
            )


    def finalize_epoch(self):
        for module_name, module_data in self._preprocessor.value.items():
            line_values : dict[str, float] = self._line_data[module_name]
            for method in self._line_aggregation:
                line_values[method] = extract_point(module_data, method)

            range_values : dict[tuple[str, str], tuple[float, float]] = self._range_data[module_name]
            for method in self._range_aggregation:
                range_values[parse_range_name(method)] = extract_range(module_data, method)

        if self._comparison_plot:
            total_sum = 0
            for module_name, module_data in self._preprocessor.value.items():
                self._comparison_data[module_name] = extract_point(module_data, self._comparison_aggregation)
                total_sum += self._comparison_data[module_name]
            for module_name in self._comparison_data:
                self._comparison_data[module_name] /= total_sum


    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        vizualizer.plot_numerical_values(
            epoch, OutputNorm.SMALL_TAG_NAME, self._line_data, self._range_data
        )

        if self._comparison_plot:
            vizualizer.plot_relations(
                epoch, self._comparison_plot_name,
                OrderedDict([(self._comparison_plot_name, self._comparison_data)])
            )

    def reset_epoch(self):
        self._preprocessor.reset()
