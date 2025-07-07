import numpy as np

from collections import OrderedDict
from typing import Iterable
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.gatherer import FeedForwardGatherer
from monitorch.preprocessor import AbstractPreprocessor, OutputActivation as OutputActivationPreprocessor
from monitorch.vizualizer import AbstractVizualizer, TagAttributes, TagType
from monitorch.numerical import extract_point

from .module_distinction import isactivation


class OutputActivation(AbstractLens):

    SMALL_TAG_NAME = "Output Activations"
    BIG_TAG_NAME = "Warning Output Activations"

    def __init__(
        self,
        inplace : bool = True,

        skip_no_grad_pass : bool = True,
        activation_only : bool = True,

        warning_plot : bool = True,

        activation_aggregation : str = 'mean',
        death_aggregation      : str = 'mean',
    ):
        assert bool(activation_aggregation)
        assert bool(death_aggregation)
        self._preprocessor = OutputActivationPreprocessor(
            death=True,
            inplace=inplace,
            record_no_grad=not skip_no_grad_pass
        )
        self._data : OrderedDict[str, dict[str, float]] = OrderedDict()

        self._activation_only = activation_only
        self._warning_plot = warning_plot
        if self._warning_plot:
            self._warning_data = {
                'worst activation_rate' : float('nan'),
                'worst death_rate'      : float('nan'),
            }


        self._gatherers = []
        self._activation_aggregation : str = activation_aggregation
        self._death_aggregation : str = death_aggregation


    def register_module(self, module : Module, module_name : str):
        if not(self._activation_only and isactivation(module)):
            return
        ffg = FeedForwardGatherer(
            module, [self._preprocessor], module_name
        )
        self._gatherers.append(ffg)
        self._data[module_name] = {}

    def detach_from_module(self):
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ does not interact with foreign preprocessor """
        pass

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        vizualizer.register_tags(
            OutputActivation.SMALL_TAG_NAME,
            TagAttributes(
                logy=False, annotate=True, big_plot=False,
                type=TagType.PROBABILITY
            )
        )
        if self._warning_plot:
            vizualizer.register_tags(
                OutputActivation.BIG_TAG_NAME,
                TagAttributes(
                    logy=False, annotate=True, big_plot=True,
                    type=TagType.PROBABILITY
                )
            )

    def finalize_epoch(self):
        worst_act = float('+inf')
        worst_death = float('-inf')

        for module_name, val_dict in self._data.items():
            activations, death = self._preprocessor.value[module_name]
            val_dict['activation_rate'] = extract_point(activations, self._activation_aggregation)
            val_dict['death_rate'] = extract_point(death, self._death_aggregation)
            worst_act   = min(worst_act,   val_dict['activation_rate'])
            worst_death = max(worst_death, val_dict['death_rate'])

        if self._warning_plot:
            self._warning_data['worst activation_rate'] = worst_act
            self._warning_data['worst death_rate'] = worst_death



    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        vizualizer.plot_probabilities(
            epoch, OutputActivation.SMALL_TAG_NAME,
            self._data
        )
        if self._warning_plot:
            vizualizer.plot_probabilities(
                epoch, OutputActivation.BIG_TAG_NAME,
                OrderedDict([(OutputActivation.BIG_TAG_NAME, self._warning_data)])
            )

    def reset_epoch(self):
        self._preprocessor.reset()
