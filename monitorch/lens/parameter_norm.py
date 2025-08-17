from collections import OrderedDict
from typing import Iterable
from .abstract_lens import AbstractLens
from torch.nn import Module

from monitorch.gatherer import EpochModuleGatherer
from monitorch.preprocessor import AbstractPreprocessor, ParameterNorm as ParameterNormPreprocessor
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.numerical import extract_point


class ParameterNorm(AbstractLens):
    """
    A lens to collect parameter norm.

    Computes L2-norm or root-mean-square on explicit lens call or epoch tick.
    Lens draws a small plot for each layer selected during initialization, optionally draws comparison plot
    between all layers.

    Parameters
    ----------
    inplace : bool = True
        Flag indicating if computation should be done in-place or in-memory.

    parameters : Iterable[str] = ('weight', 'bias')
        Parameters which the norm or rms will be computed.

    normalize_by_size : bool = False
        Flag indicating if parameter norm should be divided by root of number of elements, thus obtaining RMS of parameter.
    log_scale : bool = False
        Flag indicating if logarithmic scale should be used.

    comparison_plot : bool = True
        Flag indicating if big comparison plot should be drawn.
    aggregation_method : str = 'mean'
        Aggregation method for lines in plots.

    Examples
    --------

    Default usage is shown below.

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         ParameterNorm(),
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
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()

    To collect data more often use :meth:`collect_data`.

    >>> pnorm_lens = ParameterNorm()
    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         pnorm_lens,
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
    ...         pnorm_lens.collect_data()
    ... 
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()
    """

    def __init__(
        self,
        inplace : bool = True,

        parameters : Iterable[str] = ('weight', 'bias'),

        normalize_by_size : bool = False,
        log_scale : bool = False,

        comparison_plot : bool = True,
        aggregation_method : str = 'mean'
    ):
        self._parameters = list(parameters)
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
        if not all(hasattr(module, parameter_name) for parameter_name in self._parameters):
            return
        gatherer = EpochModuleGatherer(
            module, [self._preprocessor], module_name
        )
        self._gatherers.append(gatherer)
        for parameter_name in self._parameters:
            self._data[parameter_name][module_name] = {}

    def detach_from_module(self):
        """
        Detaches lens from module.

        Detaches gatherers and resets inner state.
        """
        for gatherer in self._gatherers:
            gatherer.detach()
        self._gatherers = []

        self._data : OrderedDict[str, OrderedDict[str, dict[str, float]]]= OrderedDict([
            (parameter_name, OrderedDict()) for parameter_name in self._parameters
        ])
        if self._comparison_plot:
            self._comparison_data : OrderedDict[str, OrderedDict[str, float]] = OrderedDict([
                (parameter_name, OrderedDict()) for parameter_name in self._parameters
            ])

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ Does not interact with foreign preprocessor. """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        For every parameter listed during initialization creates
        a small numerical plot '#PARAMETER_NAME Norm' optionally creates
        a big comparison plot  '#PARAMETER_NAME [Log] Norm Comparisson'.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
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

    def collect_data(self):
        """
        Calls gatherers to collect data.
        """
        for gatherer in self._gatherers:
            gatherer()


    def finalize_epoch(self):
        """
        Finaizes computations done through epoch.

        Aggregates parameter norms according to ``aggregation_method``
        and computes comparison values.
        """
        self.collect_data()

        for parameter_name in self._parameters:
            comparison_dict : OrderedDict[str, float]
            if self._comparison_plot:
                comparison_dict = self._comparison_data[parameter_name]
            tag_data_dict = self._data[parameter_name]
            total_sum = 1e-7
            for module_name, module_data in self._preprocessor.value.items():
                pt_val = extract_point(module_data[parameter_name], self._aggregation_method)
                tag_data_dict.setdefault(module_name, {})[self._aggregation_method] = pt_val
                total_sum += pt_val
                if self._comparison_plot:
                    comparison_dict[module_name] = pt_val

            if self._comparison_plot:
                for module_name in comparison_dict:
                    comparison_dict[module_name] /= total_sum

    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        """
        Passes computed data to visualizer.

        Passes dictionary of per layer data to '#PARAMETER_NAME Output Norm', the dictionary
        may look something like this.

        ::

            OrderedDict([
                ('lin1',   {'mean' : 0.8}, {'min' : 0.2, 'max' : 0.9}),
                ('lin2',   {'mean' : 0.6}, {'min' : 0.3, 'max' : 0.7}),
            ])

        If comparison plot needs to be plotted passes a dictionary described below to '#PARAMETER [Log] Norm Comparison'

        ::

            OrderedDict([
                ('Weight Norm Comparison', {
                    'lin1' : 0.7,
                    'lin2' : 0.3
                })
            ])

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
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
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        self._preprocessor.reset()
