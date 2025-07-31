"""
Submodule containing stateful callbacks to hook on modules.

This module implements classes for automatic data collection from PyTorch modules.
Gatherers should be used whenever there is need to keep track of data from several modules,
that can be differentiated by names.
Data aggregation and preprocessing must be done by :mod:`monitorch.preprocessor`.
The submodule provides classes to gather outputs, gradients on each pass and to pass module object to processor on call.
"""

from .abstract_gatherer import AbstractGatherer
from .feed_forward_gatherer import FeedForwardGatherer
from .backward_gatherer import BackwardGatherer
from .parameter_gradient_gatherer import ParameterGradientGatherer
from .epoch_module_gatherer import EpochModuleGatherer

__all__ = [
    "AbstractGatherer",
    "FeedForwardGatherer",
    "BackwardGatherer",
    "ParameterGradientGatherer",
    "EpochModuleGatherer",
]
