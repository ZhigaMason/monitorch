from collections import OrderedDict
from typing import Iterable
from .abstract_lens import AbstractLens

from torch.nn import Module
from monitorch.gatherer import FeedForwardGatherer
from monitorch.preprocessor import (
        AbstractPreprocessor,
        ExplicitCall,
        LossModule
)
from monitorch.visualizer import AbstractVisualizer, TagAttributes, TagType
from monitorch.numerical import extract_point, extract_range, parse_range_name

class LossMetrics(AbstractLens):
    """
    Lens to track loss and metrics.

    Tracks losses through ``push_loss`` on inspector
    or loss module object (for example ``torch.nn.MSELoss`` or ``torch.nn.NLLLoss``).
    To keep track of metrics pass metric's name during initizaliozation and use ``push_metric`` on inspector to save data.

    Strings associated with loss names are taken from inspector.

    Parameters
    ----------
    metrics : Iterable[str]|None = None
        Metrics' names to plot.
    separate_loss_and_metrics : bool = True,
        Flag indicating if loss and metric plots should be separate.

    loss_fn : Module|None  = None
        Loss function module, if provided loss values will be extract during forward pass through ``loss_fn`` object.
    loss_fn_inplace : bool = True
        Flag indicating if data from ``loss_fn`` should be aggregated inplace.

    loss_line : str|Iterable[str] = 'mean'
        Aggregation methods for loss lines.
    loss_range : str|Iterable[str]|None = 'std'
        Aggregation methods for loss bands.

    metrics_line : str|Iterable[str]       = 'mean'
        Aggregation methods for metrics' lines.
    metrics_range : str|Iterable[str]|None = None
        Aggregation methods for metrics' bands.

    Examples
    --------

    An example of training where loss is explicit pushed into inspector.

    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         LossMetrics(),
    ...     ],
    ...     module = mynet,
    ...     visualizer='matplotlib'
    ... )
    >>> 
    >>> for epoch in range(N_EPOCHS):
    ...     for data, label in train_dataloader:
    ...         optimizer.zero_grad()
    ...         prediction = mynet(data)
    ...         loss = F.binary_cross_entropy(prediction, label)
    ...         loss.backward()
    ...         optimizer.step()
    ... 
    ...         inspector.push_loss(loss.item(), train=True)
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()

    The same effect can be obtained by providind ``torch.nn`` loss object.

    >>> loss_fn = nn.BCELoss()
    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         LossMetrics(loss_fn=loss_fn),
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

    Metrics must be passed explicitly.

    >>> loss_fn = nn.BCELoss()
    >>> inspector = PyTorchInspector(
    ...     lenses = [
    ...         LossMetrics(
    ...             metrics=['accuracy'],
    ...             loss_fn=loss_fn
    ...         ),
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
    ...         accuracy_score = ((prediction > 0.5).float() == label).mean().item()
    ...         inspector.push_metric('accuracy', accuracy_score)
    ...     inspector.tick_epoch()
    >>> 
    >>> inspector.visualizer.show_fig()
    """

    def __init__(
            self, *,
            metrics : Iterable[str]|None = None,
            separate_loss_and_metrics : bool = True,

            loss_fn : Module|None  = None,
            loss_fn_inplace : bool = True,

            loss_line : str|Iterable[str]       = 'mean',
            loss_range : str|Iterable[str]|None = 'std',

            metrics_line : str|Iterable[str]       = 'mean',
            metrics_range : str|Iterable[str]|None = None,
    ):
        self._metrics : Iterable[str] = metrics if metrics else tuple()
        self._separate_loss_and_metrics = separate_loss_and_metrics
        self._call_preprocessor : ExplicitCall|None = None


        self._loss_line : list[str] = [loss_line] if isinstance(loss_line, str) else list(loss_line)
        self._loss_range : Iterable[str]
        if isinstance(loss_range, str):
            self._loss_range = [loss_range]
        elif loss_range is None:
            self._loss_range = []
        else:
            self._loss_range = loss_range

        self._metrics_line  :Iterable[str] = [metrics_line] if isinstance(metrics_line, str) else metrics_line
        self._metrics_range :Iterable[str]
        if isinstance(metrics_range, str):
            self._metrics_range = [metrics_range]
        elif metrics_range is None:
            self._metrics_range = []
        else:
            self._metrics_range = metrics_range

        self._loss_values : dict[str, float] = {}
        self._loss_ranges : dict[tuple[str, str], tuple[float, float]] = {}

        if metrics:
            self._metrics_values : dict[str, float] = {}
            self._metrics_ranges : dict[tuple[str, str], tuple[float, float]] = {}

        self._is_loss_fn = False
        if loss_fn is not None:
            self._is_loss_fn = True
            self._preprocessor = LossModule(inplace=loss_fn_inplace)
            self._loss_gatherer = FeedForwardGatherer(
                loss_fn, [self._preprocessor], 'loss'
            )

    def loss(self, *, train : bool, method : str|None = None) -> float:
        """
        Get loss from last finalization (epoch tick).

        Parameters
        ----------
        train : bool
            Flag indicating if train loss must be returned
        method : str|None = None
            Optional method of loss aggregation, default is the first provided during initialization.

        Returns
        -------
        float
            loss value

        Raises
        ------
        AttributeError
            If lens cannot get loss strings, most probably the lens was not registered with :class:`ExplicitCall` preprocessor.
        """
        if method is None:
            method = self._loss_line[0]

        if self._call_preprocessor is None:
            raise AttributeError("Cannot get loss strings.")
        loss_str = self._call_preprocessor.non_train_loss_str
        if train:
            loss_str = self._call_preprocessor.train_loss_str
        return self._loss_values[loss_str + ' ' + method]

    def register_module(self, module : Module, module_name : str):
        """ Does not interact with estimator network. """
        pass

    def detach_from_module(self):
        """
        Does not interact with estimator network. Does not detach from loss function module.
        """
        #if self._is_loss_fn:
        #    self._loss_gatherer.detach()

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """
        Saves an instance of :class:`monitorch.preprocessor.ExplicitCall`,
        other predprocessors are ignored.

        Parameters
        ----------
        ext_ppr : AbstractPreprocessor
            External preprocessor to register (or ignore).
        """
        if isinstance(ext_ppr, ExplicitCall):
            self._call_preprocessor = ext_ppr
            if self._is_loss_fn:
                self._preprocessor.set_loss_strs( # duck polymorphism, should be extracted to be atleast class polymorphism
                    ext_ppr.train_loss_str,
                    ext_ppr.non_train_loss_str,

                )

    def introduce_tags(self, vizualizer : AbstractVisualizer):
        """
        Introduces lens's plots to visualizer.

        Registers one big tag if ``separate_loss_and_metrics`` is ``False`` named ``'Loss & Metrics'``.
        If ``separate_loss_and_metrics`` is ``True``, registers two big tags: ``'Loss'`` and ``'Metrics'``.

        Parameters
        ----------
        visualzier : AbstractVisualizer
            A visualizer object to pass tag attributes to.
        """
        if self._separate_loss_and_metrics:
            vizualizer.register_tags('Loss',    TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))
            if self._metrics:
                vizualizer.register_tags('Metrics', TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))
        else:
            vizualizer.register_tags('Loss & Metrics',    TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))

    def finalize_epoch(self):
        """
        Finalizes loss and metrics computation.
        """
        self._finalize_loss()
        if self._metrics:
            self._finalize_metrics()

    def _finalize_loss(self):
        assert self._call_preprocessor is not None

        train_loss_str = self._call_preprocessor.train_loss_str
        non_train_loss_str = self._call_preprocessor.non_train_loss_str

        raw_train_loss = None
        raw_non_train_loss = None

        if self._is_loss_fn:
            raw_train_loss = self._preprocessor.value[train_loss_str]
            raw_non_train_loss = self._preprocessor.value.get(non_train_loss_str, False)
        else:
            raw_train_loss = self._call_preprocessor.value[train_loss_str]
            raw_non_train_loss = self._call_preprocessor.value.get(non_train_loss_str, False)

        if not raw_non_train_loss:
            raw_non_train_loss = None

        # line aggregation
        for loss_line in self._loss_line:
            pt = extract_point(raw_train_loss, loss_line)
            self._loss_values[train_loss_str + ' ' + loss_line] = pt
            if raw_non_train_loss is not None:
                pt = extract_point(raw_non_train_loss, loss_line)
                self._loss_values[non_train_loss_str + ' ' + loss_line] = pt

        # range aggreagation
        for loss_range in self._loss_range:
            range_tuple = extract_range(raw_train_loss, loss_range)
            lo_name, up_name = parse_range_name(loss_range)
            self._loss_ranges[(train_loss_str + ' ' + lo_name, train_loss_str + ' ' + up_name)] = range_tuple
            if raw_non_train_loss is not None:
                range_tuple = extract_range(raw_non_train_loss, loss_range)
                self._loss_ranges[(non_train_loss_str + ' ' + lo_name, non_train_loss_str + ' ' + up_name)] = range_tuple


    def _finalize_metrics(self):
        assert self._call_preprocessor is not None
        for metric in self._metrics:
            raw_val = self._call_preprocessor.value[metric]
            # line aggregation
            for agg_line in self._metrics_line:
                pt = extract_point(raw_val, agg_line)
                self._metrics_values[metric + ' ' + agg_line] = pt
            # range aggreagation
            for agg_range in self._metrics_ranges:
                range_tuple = extract_range(raw_val, agg_range)
                lo_name, up_name = parse_range_name(agg_range)
                self._metrics_ranges[(metric + ' ' + lo_name, metric + ' ' + up_name)] = range_tuple

    def vizualize(self, vizualizer : AbstractVisualizer, epoch : int):
        """
        Visualizes loss and metrics.

        Passes loss Ordered dictionaries that may look like this
        ::

            OrderedDict([
                ('Loss', {'train_loss mean' : 0.89, 'val_loss mean' : 0.98})
            ])

            OrderedDict([
                ('Loss', {
                    ('train_loss Q1', 'train_loss Q3') : (0.79, 0.99),
                    ('val_loss Q1',   'val_loss Q3')   : (0.90, 1.06),
                })
            ])

        Metrics dictionaries are similar, though may use other top level index (i.e. 'Metrics' instead of 'Loss').

        Parameters
        ----------
        visualizer : AbstractVisualizer
            The visualizer object responsbile for drawing plots.
        epoch : int
            Computation's epoch number.
        """
        assert self._call_preprocessor is not None
        loss_tag, metrics_tag = 'Loss', 'Metrics'
        if not self._separate_loss_and_metrics:
            loss_tag = metrics_tag = 'Loss & Metrics'


        vizualizer.plot_numerical_values(
            epoch, loss_tag,
            OrderedDict([(loss_tag, self._loss_values)]),
            OrderedDict([(loss_tag, self._loss_ranges)])
        )

        if self._metrics:
            vizualizer.plot_numerical_values(
                epoch, metrics_tag,
                OrderedDict([(metrics_tag, self._metrics_values)]),
                OrderedDict([(metrics_tag, self._metrics_ranges)])
            )

    def reset_epoch(self):
        """
        Resets inner state.

        Resets data computed during last epoch and resets preprocessors.
        """
        if self._is_loss_fn:
            self._preprocessor.reset()
