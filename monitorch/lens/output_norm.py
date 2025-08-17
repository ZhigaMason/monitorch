from torch.nn import Module
from collections import OrderedDict
from typing import Iterable, Type

from monitorch.preprocessor import AbstractPreprocessor, OutputNorm as OutputNormPreprocessor
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.gatherer import FeedForwardGatherer
from monitorch.numerical import extract_point, extract_range, parse_range_name



from .module_distinction import isactivation
from .abstract_lens import AbstractLens

class OutputNorm(AbstractLens):
    """
    A lens to examine norm of layer outputs.

    Computes L2-norm or root-mean-square of outputs produced during forward pass through module.
    Lens draws a small plot for each layer selected during initialization, optionally draws comparison plot
    between all layers.

    Parameters
    ----------
    inplace : bool = True
        Flag indicating if computation should be done in-place or in-memory.

    skip_no_grad_pass : bool = True
        Flag indicating if data collected during ``torch.no_grad`` should be ignored.
        It is expected that those passes are either validation or prediction,
        and are no relevant to network's learning.

    normalize_by_size : bool = False
        Flag indicating if output norm should be divided by root of number of elements, thus obtaining RMS of output.
    log_scale : bool = False
        Flag indicating if logarithmic scale should be used.

    activation : bool = True
        Flag indicating if activation function layers' data should be collected and displayed.

    include : Iterable[Type[Module]] = tuple()
        Additional layer types to include for inspection.
    exclude : Iterable[Type[Module]] = tuple()
        Layer types to exclude from expection.
        Overrides all settings.

    comparison_plot : bool = True
        Flag indicating if big comparison plot should be drawn.
    comparison_aggregation : str|None = None
        Epoch level aggregation used on every output sequence for comparison plot.
        Default is the same as the first ``line_aggregation``.

    line_aggregation : str|Iterable[str] = 'mean'
        Aggregation method for lines in plots.
    range_aggregation : str|Iterable[str]|None = ('std', 'min-max')
        Aggregation method for bands in plots.

    Examples
    --------

    Default usage with no-grad validation

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         OutputNorm(),
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

    _SMALL_TAG_NAME = "Output Norm"

    def __init__(
            self,
            inplace : bool = True,
            skip_no_grad_pass : bool = True,
            normalize_by_size : bool = False,
            log_scale : bool = False,

            activation : bool = True,

            include : Iterable[Type[Module]] = tuple(),
            exclude : Iterable[Type[Module]] = tuple(),

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

        self._activation = activation
        self._include = include
        self._exclude = exclude

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
        """
        Registers (or ignores) module.

        Registers modules guided by ``activation`` flag set during initialization
        and includes all modules of types mentioned in ``include``.
        Exclusion by ``exclude`` parameter overrides every other configuration.

        Parameters
        ----------
        module : torch.nn.Module
            The module object to hook gatherers onto.
        module_name : str
            Name of the module, module's information will be passed to visaulizer under this name.
        """
        if module.__class__ not in self._exclude and ((self._activation and isactivation(module)) or module.__class__ in self._include):
            ffg = FeedForwardGatherer(
                module, [self._preprocessor], module_name
            )
            self._gatherers.append(ffg)
            self._line_data[module_name]  = {}
            self._range_data[module_name] = {}

    def detach_from_module(self):
        """
        Detaches lens from module.

        Detaches gatherers and resets inner state.
        """
        for ffg in self._gatherers:
            ffg.detach()
        self._gatherers = []

        self._line_data  : OrderedDict[str, dict[str, float]] = OrderedDict()
        self._range_data : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]] = OrderedDict()
        if self._comparison_plot:
            self._comparison_data : OrderedDict[str, float]= OrderedDict()

    def register_foreign_preprocessor(self, _ : AbstractPreprocessor):
        """ Does not interact with foreign preprocessor. """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        Intorduces one small plot 'Output Norm', where per layer data is plotted, its type is ``NUMERICAL``.
        If ``comparison_plot`` is ``True`` also registers a big ``RELATIONS`` plot
        '#AGGREGATION_METHOD Output [Log] Norm Comparison' tweaked by initialization parameters.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
        vizualizer.register_tags(
            OutputNorm._SMALL_TAG_NAME, self._small_tag_attr
        )
        if self._comparison_plot:
            vizualizer.register_tags(
                self._comparison_plot_name,
                TagAttributes(
                    logy=False, big_plot=True, annotate=False, type=TagType.RELATIONS
                )
            )


    def finalize_epoch(self):
        """
        Finaizes computations done through epoch.

        Aggregates parameter gradient norms and optionally inner product according to ``line_aggregation`` and ``range_aggregation``.
        """
        for module_name, module_data in self._preprocessor.value.items():
            line_values : dict[str, float] = self._line_data[module_name]
            for method in self._line_aggregation:
                line_values[method] = extract_point(module_data, method)

            range_values : dict[tuple[str, str], tuple[float, float]] = self._range_data[module_name]
            for method in self._range_aggregation:
                range_values[parse_range_name(method)] = extract_range(module_data, method)

        if self._comparison_plot:
            total_sum = 1e-7
            for module_name, module_data in self._preprocessor.value.items():
                self._comparison_data[module_name] = extract_point(module_data, self._comparison_aggregation)
                total_sum += self._comparison_data[module_name]
            for module_name in self._comparison_data:
                self._comparison_data[module_name] /= total_sum


    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        """
        Passes computed data to visualizer.

        Passes dictionary of per layer data to 'Output Norm', the dictionary
        may look something like this.

        ::

            OrderedDict([
                ('relu1',   {'mean' : 0.8}, {'min' : 0.2, 'max' : 0.9}),
                ('relu2',   {'mean' : 0.6}, {'min' : 0.3, 'max' : 0.7}),
            ])

        If comparison plot needs to be plotted passes a dictionary described
        below to '#AGGREGATION_METHOD Output [Log] Norm Comparison'

        ::

            OrderedDict([
                ('Mean Output Log Norm Comparison', {
                    'relu1' : 0.7,
                    'relu2' : 0.3
                })
            ])

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
        vizualizer.plot_numerical_values(
            epoch, OutputNorm._SMALL_TAG_NAME, self._line_data, self._range_data
        )

        if self._comparison_plot:
            vizualizer.plot_relations(
                epoch, self._comparison_plot_name,
                OrderedDict([(self._comparison_plot_name, self._comparison_data)])
            )

    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        self._preprocessor.reset()
