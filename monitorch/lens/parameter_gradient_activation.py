from typing import Iterable
from collections import OrderedDict
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor, GradientActivation
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.gatherer import ParameterGradientGatherer
from monitorch.numerical import extract_point


class ParameterGradientActivation(AbstractLens):
    """
    A lens to inspect neuron activation through parameter gradients.

    Neuron is active if its gradient is non-zero, if neuron is inactive for the whole batch iteration,
    we say that a neuron is dead. Activation rate in an epoch is a measure of layers entropy (high activation - high entropy),
    while death rate is a measure of overcapacity, because some neurons are not used.

    This lens lets you investigate those values.
    In addition it allows to plot worst activation and death rates accross the whole model into one big warning plot.

    Parameters
    ----------
    inplace : bool = True
        Flag indicating if computation should be done in-place or in-memory.

    parameters : str|Iterable[str] = ('weight', 'bias')
        Parameters which gradient will be studied.

    warning_plot : bool = True
        Flag indicating if big warning plot should be added.

    activation_aggregation : str = 'mean'
        Aggregation method used to collect activation rate.
    death_aggregation      : str = 'mean'
        Aggregation method used to collect death rate.

    Examples
    --------

    Default usage is as simple as just mentioning ``ParameterGradientActivation`` to inspector.

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         ParameterGradientActivation(),
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
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()
    """

    _BIG_TAG_NAME = "Warning Gradient Activations"

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
        """
        Registers (or ignores) module.

        Registers any module that has all of the parameters listed during initialization.

        Parameters
        ----------
        module : torch.nn.Module
            The module object to hook gatherers onto.
        module_name : str
            Name of the module, module's information will be passed to visaulizer under this name.
        """
        if not all(hasattr(module, parameter_name) for parameter_name in self._preprocessors):
            return

        for parameter, preprocessor in self._preprocessors.items():
            pgg = ParameterGradientGatherer(
                parameter,
                module, [preprocessor], module_name
            )
            self._gatherers.append(pgg)

    def detach_from_module(self):
        """
        Detaches lens from module.

        Detaches gatherers and resets inner state.
        """
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []
        self._data : dict[str, OrderedDict[str, dict[str, float]]] = {
                parameter_name:OrderedDict()
                for parameter_name in self._data.keys()
        }
        if self._warning_plot:
            self._warning_data = {}

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ Does not interact with foreign preprocessor. """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        For every parameter creates a small probability tag '#PARAMETER_NAME Gradient Activation',
        if warning plot is on, also adds a big warning probability plot for every parameter.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
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
                ParameterGradientActivation._BIG_TAG_NAME,
                TagAttributes(
                    logy=False,
                    big_plot=True,
                    annotate=True,
                    type=TagType.PROBABILITY
                )
            )

    def finalize_epoch(self):
        """
        Finalizes computations done thorugh epoch.

        Aggregates activations and death rates according to ``activation_aggregation`` and ``death_aggregation``
        and computes worst activation (minimal) and worst death rates (maximal) for every parameter.
        """
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
        """
        Passes computed data to visualizer.

        For every parameter listed during initialization.
        Passes dictionary of per layer data to '#PARAMETER_NAME Gradient Activations', the dictionary
        may look something like this.
        ::

            OrderedDict([
                ('lin1', {'activation_rate' : 0.8, 'death_rate' : 0.3}),
                ('lin2', {'activation_rate' : 0.5, 'death_rate' : 0.3}),
            ])

        If warning plot needs to be plotted passes a dictionary described below to 'Warning #PARAMETER_NAME Gradient Activation'
        ::

            OrderedDict([
                ('Warning Weight Gradient Activation', {
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
        for parameter_name in self._preprocessors:
            vizualizer.plot_probabilities(
                epoch, f"{parameter_name} Gradient Activation".title(),
                self._data[parameter_name]
            )
        if self._warning_plot:
            vizualizer.plot_probabilities(
                epoch, ParameterGradientActivation._BIG_TAG_NAME,
                OrderedDict([(ParameterGradientActivation._BIG_TAG_NAME, self._warning_data)])
            )

    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        for preprocessor in self._preprocessors.values():
            preprocessor.reset()
