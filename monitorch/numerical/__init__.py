"""
    Submodule for generic numerical computations
"""

from .RunningValue import RunningMeanVar, RunningValue, extract_point, extract_range, parse_range_name
from .ActivationComputation import reduce_activation_to_activation_rates
from .threshold_for_module import threshold_for_module
