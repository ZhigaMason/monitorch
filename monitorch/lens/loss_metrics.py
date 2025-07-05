import numpy as np
from collections import OrderedDict
from typing import Iterable
from .abstract_lens import AbstractLens

from torch.nn import Module
from monitorch.gatherer import FeedForwardGatherer
from monitorch.preprocessor import (
        AbstractPreprocessor,
        ExplicitCall,
        LossModuleRunning, LossModuleMemory
)
from monitorch.vizualizer import AbstractVizualizer, TagAttributes, TagType
from monitorch.numerical import extract_point, extract_range, parse_range_name

class LossMetrics(AbstractLens):

    def __init__(
            self, *,
            loss : bool = True,
            metrics : Iterable[str]|None = None,
            separate_loss_and_metrics : bool = True,

            loss_fn : Module|None  = None,
            loss_fn_inplace : bool = True,

            loss_line : str       = 'mean',
            loss_range : str|None = 'std',

            metrics_line : str       = 'mean',
            metrics_range : str|None = None,
    ):
        self._loss = loss
        self._metrics : Iterable[str] = metrics if metrics else tuple()
        self._separate_loss_and_metrics = separate_loss_and_metrics
        self._call_preprocessor : ExplicitCall|None = None

        self._loss_line  = loss_line
        self._loss_range = loss_range

        self._metrics_line  = metrics_line
        self._metrics_range = metrics_range

        if loss:
            self._loss_values : dict[str, float] = {}
            self._loss_ranges : dict[tuple[str, str], tuple[float, float]] = {}

        if metrics:
            self._metrics_values : dict[str, float] = {}
            self._metrics_ranges : dict[tuple[str, str], tuple[float, float]] = {}

        self._is_loss_fn = False
        if loss_fn is not None:
            self._is_loss_fn = True
            self._preprocessor = LossModuleRunning() if loss_fn_inplace else LossModuleMemory()
            self._loss_gatherer = FeedForwardGatherer(
                loss_fn, [self._preprocessor], 'loss'
            )

    def register_module(self, module : Module, module_name : str):
        pass

    def detach_from_module(self):
        if self._is_loss_fn:
            self._loss_gatherer.detach()

    def register_foreign_preprocessor(self, ext_ppr : AbstractPreprocessor):
        """ gets a reference of foreign preprocessor to collect data from """
        if isinstance(ext_ppr, ExplicitCall):
            self._call_preprocessor = ext_ppr
            if self._is_loss_fn:
                self._preprocessor.set_loss_strs( # duck polymorphism, should be extracted to be atleast class polymorphism
                    ext_ppr.train_loss_str,
                    ext_ppr.non_train_loss_str,

                )

    def introduce_tags(self, vizualizer : AbstractVizualizer):
        if self._separate_loss_and_metrics:
            if self._loss:
                vizualizer.register_tags('Loss',    TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))
            if self._metrics:
                vizualizer.register_tags('Metrics', TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))
        else:
            if self._loss or self._metrics:
                vizualizer.register_tags('Loss & Metrics',    TagAttributes(logy=False, big_plot=True, annotate=True, type=TagType.NUMERICAL))

    def finalize_epoch(self):
        if self._loss:
            self._finalize_loss()
        if self._metrics:
            self._finalize_metrics()

    def _finalize_loss(self):
        assert self._call_preprocessor is not None

        train_loss_str = self._call_preprocessor.train_loss_str
        non_train_loss_str = self._call_preprocessor.non_train_loss_str
        lo_name, up_name = parse_range_name(self._loss_range)

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

        pt = extract_point(raw_train_loss, self._loss_line)
        range_tuple = extract_range(raw_train_loss, self._loss_range)

        self._loss_values[train_loss_str + ' ' + self._loss_line] = pt
        self._loss_ranges[(train_loss_str + ' ' + lo_name, train_loss_str + ' ' + up_name)] = range_tuple
        if raw_non_train_loss is not None:
            pt = extract_point(raw_non_train_loss, self._loss_line)
            range_tuple = extract_range(raw_non_train_loss, self._loss_range)

            self._loss_values[non_train_loss_str + ' ' + self._loss_line] = pt
            self._loss_ranges[(non_train_loss_str + ' ' + lo_name, non_train_loss_str + ' ' + up_name)] = range_tuple


    def _finalize_metrics(self):
        assert self._call_preprocessor is not None
        for metric in self._metrics:
            raw_val = self._call_preprocessor.value[metric]
            pt = extract_point(raw_val, self._metrics_line)
            self._metrics_values[metric + ' ' + self._metrics_line] = pt
            if self._metrics_range:
                lo_name, up_name = parse_range_name(self._metrics_range)
                range_tuple = extract_range(raw_val, self._metrics_range)
                self._metrics_ranges[(metric + ' ' + lo_name, metric + ' ' + up_name)] = range_tuple

    def vizualize(self, vizualizer : AbstractVizualizer, epoch : int):
        assert self._call_preprocessor is not None
        loss_tag, metrics_tag = 'Loss', 'Metrics'
        if not self._separate_loss_and_metrics:
            loss_tag = metrics_tag = 'Loss & Metrics'


        if self._loss:
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
        if self._is_loss_fn:
            self._preprocessor.reset()
