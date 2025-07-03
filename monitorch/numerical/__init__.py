"""
    Submodule for generic numerical computations
"""

from .RunningValue import RunningMeanVar, RunningValue
from .ActivationComputation import reduce_activation_to_activation_rates
from .threshold_for_module import threshold_for_module
