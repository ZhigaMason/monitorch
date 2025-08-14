import numpy as np

from collections import OrderedDict
from typing import Iterable, Type
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.gatherer import FeedForwardGatherer
from monitorch.preprocessor import AbstractPreprocessor, OutputActivation as OutputActivationPreprocessor
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.numerical import extract_point

from .module_distinction import isactivation, isdropout


class OutputActivation(AbstractLens):
    """
    A lens to inspect neuron activation through output.

    Neuron is active if it yields non-zero value, if neuron is inactive for the whole batch iteration,
    we say that a neuron is dead. Activation rate in an epoch is a measure of layers entropy (high activation - high entropy),
    while death rate is a measure of overcapacity, because some neurons are not used.

    This lens lets you investigate those values.
    In addition it allows to plot worst activation and death rates accross the whole model into one big warning plot.

    Parameters
    ----------
    inplace : bool = True
        Flag indicating if computation should be done in-place or in-memory.

    skip_no_grad_pass : bool = True
        Flag indicating if data collected during ``torch.no_grad`` should be ignored.
        It is expected that those passes are either validation or prediction,
        and are no relevant to network's learning.

    activation : bool = True
        Flag indicating if activation function layers' data should be collected and displayed.
    dropout : bool = True,
        Flag indicating if dropout layers' data should be collected and displayed.

    include : Iterable[Type[Module]] = tuple()
        Additional layer types to include for inspection.
    exclude : Iterable[Type[Module]] = tuple()
        Layer types to exclude from expection.
        Overrides all settings.

    warning_plot : bool = True
        Flag indicating if big warning plot should be added.

    activation_aggregation : str = 'mean'
        Aggregation method used to collect activation rate.
    death_aggregation      : str = 'mean'
        Aggregation method used to collect death rate.

    Examples
    --------

    Default usage with no-grad validation

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         OutputActivation(),
    ...     ],
    ...     module = mynet,
    ...     visualizer='matplotlib'
    ... )
    >>> 
    >>> for epoch in range(N_EPOCHS):
    ...     for data, label in train_dataloader:
    ...         optimizer.zero_grad()
    ...         prediction = mynet(data)
    ...         loss = loss_fn(prediction, label)
    ...         loss.backward()
    ...         optimizer.step()
    ... 
    ...     with torch.no_grad(): # outputs inside this block are not recorded
    ...         for data, label in val_dataloader:
    ...             prediction = mynet(data)
    ...             loss = loss_fn(prediction, label)
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()

    """

    _SMALL_TAG_NAME = "Output Activations"
    _BIG_TAG_NAME = "Warning Output Activations"

    def __init__(
        self,
        inplace : bool = True,

        skip_no_grad_pass : bool = True,

        activation : bool = True,
        dropout : bool = True,
        include : Iterable[Type[Module]] = tuple(),
        exclude : Iterable[Type[Module]] = tuple(),

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

        self._activation = activation
        self._dropout = dropout
        self._include = include
        self._exclude = exclude


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
        """
        Registers (or ignores) module.

        Registers modules guided by ``activation`` and ``dropout`` flags during initialization
        and includes all modules of types mentioned in ``include``.
        Exclusion by ``exclude`` parameter overrides every other configuration.

        Parameters
        ----------
        module : torch.nn.Module
            The module object to hook gatherers onto.
        module_name : str
            Name of the module, module's information will be passed to visaulizer under this name.
        """
        if module.__class__ in self._exclude or (
            not (module.__class__ in self._include) and
            not (self._activation and isactivation(module)) and
            not (self._dropout and isdropout(module))
        ):
            return
        ffg = FeedForwardGatherer(
            module, [self._preprocessor], module_name
        )
        self._gatherers.append(ffg)
        self._data[module_name] = {}

    def detach_from_module(self):
        """
        Detaches lens from module.

        Detaches gatherers and resets inner state.
        """
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []
        self._data = OrderedDict()
        if self._warning_plot:
            self._warning_data = {
                'worst activation_rate' : float('nan'),
                'worst death_rate'      : float('nan'),
            }

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ Does not interact with foreign preprocessor. """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        Intorduces one small plot 'Output Activations',
        where per layer data is plotted, its type is ``PROBABILITY``.
        If ``warning_plot`` is ``True`` also
        registers a big ``PROBABILITY`` plot 'Warning Output Activations'.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
        vizualizer.register_tags(
            OutputActivation._SMALL_TAG_NAME,
            TagAttributes(
                logy=False, annotate=True, big_plot=False,
                type=TagType.PROBABILITY
            )
        )
        if self._warning_plot:
            vizualizer.register_tags(
                OutputActivation._BIG_TAG_NAME,
                TagAttributes(
                    logy=False, annotate=True, big_plot=True,
                    type=TagType.PROBABILITY
                )
            )

    def finalize_epoch(self):
        """
        Finaizes computations done through epoch.

        Aggregates activations and death rates according to ``activation_aggregation`` and ``death_aggregation``
        and computes worst activation (minimal) and worst death rates (maximal).
        """
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



    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        """
        Passes computed data to visualizer.

        Passes dictionary of per layer data to 'Output Activations', the dictionary
        may look something like this.

        ::

            OrderedDict([
                ('relu1',   {'activation_rate' : 0.8, 'death_rate' : 0.3}),
                ('dropout', {'activation_rate' : 0.9, 'death_rate' : 0.3}),
                ('relu2',   {'activation_rate' : 0.2, 'death_rate' : 0.1})
            ])

        If warning plot needs to be plotted passes a dictionary described below to 'Warning Output Activation'

        ::

            OrderedDict([
                ('Warning Output Activation', {
                    'worst activation_rate' : 0.2,
                    'worst death_rate'      : 0.3
                })
            ])

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
        vizualizer.plot_probabilities(
            epoch, OutputActivation._SMALL_TAG_NAME,
            self._data
        )
        if self._warning_plot:
            vizualizer.plot_probabilities(
                epoch, OutputActivation._BIG_TAG_NAME,
                OrderedDict([(OutputActivation._BIG_TAG_NAME, self._warning_data)])
            )

    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        self._preprocessor.reset()
