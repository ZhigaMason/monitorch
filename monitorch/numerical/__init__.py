"""
numerical
===================
Submodule for generic numerical computation.

This subpackage provides function to reduce activation tensor's space dimensions,
running statistics computation, functions to extract describtive numerical values from lists and runnning statistics,
as well as utilities to parse string descriptions of these statistics.

Examples
--------
>>> from monitorch.numerical import RunningMeanVar, extract_point
>>> extract_point([4, 5, 6], 'mean')
5.0
>>> rmv = RunningMeanVar()
>>> rmv.append(4)
>>> rmv.update(5)
>>> rmv.update(6)
>>> extract_point(rmv, 'mean')
5.0
"""

from .RunningValue import RunningMeanVar, RunningValue, extract_point, extract_range, parse_range_name
from .ActivationComputation import reduce_activation_to_activation_rates

__all__ = [
    "RunningMeanVar",
    "extract_point",
    "extract_range",
    "parse_range_name",
    "reduce_activation_to_activation_rates"
]
