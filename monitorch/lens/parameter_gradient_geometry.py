from typing import Iterable
from collections import OrderedDict
from .abstract_lens import AbstractLens
from torch.nn import Module
from monitorch.preprocessor import AbstractPreprocessor, GradientGeometry
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.gatherer import ParameterGradientGatherer
from monitorch.numerical import extract_point, extract_range, parse_range_name


class ParameterGradientGeometry(AbstractLens):
    """
    Lens to examine geometry of gradients with respect to parameters.

    Computes L2-norm or root-mean-square of gradients on every backward pass through parameter.
    Optionally computes normalized inner product between gradients from two consecutive backward passes.

    Computing inner product requires gradients from both epochs, hence the gradient will be saved after the computation is finished.
    It drives space consumption linearly by size of studied parameters.

    Parameters
    ----------
    inplace : bool = True
        Flag indicating if computation should be done in-place or in-memory.

    normalize_by_size : bool = False
        Flag indicating if output norm should be divided by root of number of elements, thus obtaining RMS of output.
    log_scale : bool = False
        Flag indicating if logarithmic scale should be used.

    compute_adj_prod : bool = True
        Flag indicating if inner product normalized by L2-norm
        between gradients from consecutive backward passes hould be computed.

    parameters : str|Iterable[str] = ('weight', 'bias')
        Parameters which gradient will be studied.

    line_aggregation : str|Iterable[str] = 'mean'
        Aggregation method for lines in plots.
    range_aggregation : str|Iterable[str]|None = ('std', 'min-max')
        Aggregation method for bands in plots.

    Examples
    --------

    Default usage is shown below.

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         ParameterGradientGeometry(),
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
    """

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
        self._compute_adj_prod = compute_adj_prod
        self._preprocessors = OrderedDict([
            (parameter, GradientGeometry(inplace=inplace, normalize=normalize_by_size, adj_prod=compute_adj_prod))
            for parameter in parameters
        ])
        self._gatherers = []
        self._line_data : dict[str, OrderedDict[str, dict[str, float]]] = {}
        self._range_data : dict[str, OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]] = {}
        if self._compute_adj_prod:
            self._line_adj_prod_data : dict[str, OrderedDict[str, dict[str, float]]] = {}
            self._range_adj_prod_data : dict[str, OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]] = {}
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

        self._line_data : dict[str, OrderedDict[str, dict[str, float]]] = {}
        self._range_data : dict[str, OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]] = {}
        if self._compute_adj_prod:
            self._line_adj_prod_data : dict[str, OrderedDict[str, dict[str, float]]] = {}
            self._range_adj_prod_data : dict[str, OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]] = {}

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ Does not interact with foreign preprocessor. """
        pass

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        For every parameter listed during initialization creates
        a small numerical plot '#PARAMETER_NAME Gradient Norm' optionally creates
        a big comparison plot  '#PARAMETER_NAME Gradient Adjacent Prod'.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
        for parameter_name in self._preprocessors:
            vizualizer.register_tags(
                f"{parameter_name} Gradient Norm".title(),
                TagAttributes(
                    logy=self._log_scale,
                    big_plot=False,
                    annotate=True,
                    type=TagType.NUMERICAL
                )
            )
            if self._compute_adj_prod:
                vizualizer.register_tags(
                    f"{parameter_name} Gradient Adjacent Prod".title(),
                    TagAttributes(
                        logy=False,
                        big_plot=False,
                        annotate=True,
                        type=TagType.NUMERICAL,
                        ylim=(-1, 1)
                    )
                )

    def finalize_epoch(self):
        """
        Finaizes computations done through epoch.

        Aggregates parameter gradient norms and optionally inner product according to ``line_aggregation`` and ``range_aggregation``.
        """
        for parameter_name, preprocessor in self._preprocessors.items():
            line_norm_tag_dict : OrderedDict[str, dict[str, float]] = self._line_data.setdefault(parameter_name, OrderedDict())
            range_norm_tag_dict : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]] = self._range_data.setdefault(parameter_name, OrderedDict())
            line_prod_tag_dict : OrderedDict[str, dict[str, float]]
            range_prod_tag_dict : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]
            if self._compute_adj_prod:
                line_prod_tag_dict = self._line_adj_prod_data.setdefault(parameter_name, OrderedDict())
                range_prod_tag_dict = self._range_adj_prod_data.setdefault(parameter_name, OrderedDict())
            for module_name, value in preprocessor.value.items():
                line_norm_dict  : dict[str, float] = line_norm_tag_dict.setdefault(module_name, {})
                range_norm_dict : dict[tuple[str, str], tuple[float, float]]= range_norm_tag_dict.setdefault(module_name, {})
                line_prod_dict  : dict[str, float]
                range_prod_dict : dict[tuple[str, str], tuple[float, float]]
                if self._compute_adj_prod:
                    line_prod_dict = line_prod_tag_dict.setdefault(module_name, {})
                    range_prod_dict = range_prod_tag_dict.setdefault(module_name, {})

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
            self._line_data[parameter_name] = OrderedDict(reversed(line_norm_tag_dict.items()))
            self._range_data[parameter_name] = OrderedDict(reversed(range_norm_tag_dict.items()))
            if self._compute_adj_prod:
                self._line_adj_prod_data[parameter_name] = OrderedDict(reversed(line_prod_tag_dict.items()))
                self._range_adj_prod_data[parameter_name] = OrderedDict(reversed(range_prod_tag_dict.items()))



    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        """
        Passes computed data to visualizer.

        Passes dictionary of per layer data to '#PARAMETER_NAME Gradient Norm', the dictionary
        may look something like this.

        ::

            OrderedDict([
                ('lin1',   {'mean' : 0.8}, {'min' : 0.2, 'max' : 0.9}),
                ('lin2',   {'mean' : 0.6}, {'min' : 0.3, 'max' : 0.7}),
            ])

        Gradient adjacent product dictionary looks the same.

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
        for parameter_name in self._preprocessors:
            vizualizer.plot_numerical_values(
                epoch, f"{parameter_name} Gradient Norm".title(),
                self._line_data[parameter_name], self._range_data[parameter_name]
            )
            if self._compute_adj_prod:
                vizualizer.plot_numerical_values(
                    epoch, f"{parameter_name} Gradient Adjacent Prod".title(),
                    self._line_adj_prod_data[parameter_name], self._range_adj_prod_data[parameter_name]
                )

    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        for preprocessor in self._preprocessors.values():
            preprocessor.reset()
