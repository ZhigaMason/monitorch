from typing import Iterable
from collections import OrderedDict
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor, GradientActivation
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.gatherer import ParameterGradientGatherer
from monitorch.numerical import extract_point


class ParameterGradientActivation(AbstractLens):

    BIG_TAG_NAME = "Warning Gradient Activations"

    def __init__(
        self,
        inplace : bool = True,

        warning_plot : bool = True,
        parameters : str|Iterable[str] = ('weight', 'bias'),

        activation_aggregation : str = 'mean',
        death_aggregation      : str = 'mean',
    ):
        self._warning_plot = warning_plot
        if warning_plot:
            self._warning_data = {}
        self._preprocessors = OrderedDict([
            (parameter, GradientActivation(inplace=inplace, death=True))
            for parameter in parameters
        ])
        self._gatherers = []

        self._data : dict[str, OrderedDict[str, dict[str, float]]] = {
                parameter_name:OrderedDict()
                for parameter_name in parameters
        }


        self._activation_aggregation = activation_aggregation
        self._death_aggregation = death_aggregation

    def register_module(self, module : Module, module_name : str):
        if not all(hasattr(module, parameter_name) for parameter_name in self._preprocessors):
            return

        for parameter, preprocessor in self._preprocessors.items():
            pgg = ParameterGradientGatherer(
                parameter,
                module, [preprocessor], module_name
            )
            self._gatherers.append(pgg)

    def detach_from_module(self):
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ no external data collection """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        for parameter_name in self._preprocessors:
            vizualizer.register_tags(
                f"{parameter_name} Gradient Activation".title(),
                TagAttributes(
                    logy=False,
                    big_plot=False,
                    annotate=True,
                    type=TagType.PROBABILITY
                )
            )
        if self._warning_plot:
            vizualizer.register_tags(
                ParameterGradientActivation.BIG_TAG_NAME,
                TagAttributes(
                    logy=False,
                    big_plot=True,
                    annotate=True,
                    type=TagType.PROBABILITY
                )
            )

    def finalize_epoch(self):
        worst_act = float('+inf')
        worst_death = float('-inf')
        for parameter_name, preprocessor in self._preprocessors.items():
            tag_dict = self._data.setdefault(parameter_name, OrderedDict())
            for module_name, (act_rate, death) in preprocessor.value.items():
                val_dict = tag_dict.setdefault(module_name, {})
                val_dict['activation_rate'] = extract_point(act_rate, self._activation_aggregation)
                val_dict['death_rate'] = extract_point(death, self._death_aggregation)
                worst_act   = min(worst_act,   val_dict['activation_rate'])
                worst_death = max(worst_death, val_dict['death_rate'])
            self._data[parameter_name] = OrderedDict(reversed(tag_dict.items()))

        if self._warning_plot:
            self._warning_data['worst activation_rate'] = worst_act
            self._warning_data['worst death_rate'] = worst_death


    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        for parameter_name in self._preprocessors:
            vizualizer.plot_probabilities(
                epoch, f"{parameter_name} Gradient Activation".title(),
                self._data[parameter_name]
            )
        if self._warning_plot:
            vizualizer.plot_probabilities(
                epoch, ParameterGradientActivation.BIG_TAG_NAME,
                OrderedDict([(ParameterGradientActivation.BIG_TAG_NAME, self._warning_data)])
            )

    def reset_epoch(self):
        for preprocessor in self._preprocessors.values():
            preprocessor.reset()
