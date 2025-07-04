"""
    A file containing utility classes used to record running values
"""

import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class RunningMeanVar:
    count : int   = 0
    mean  : float = 0
    var   : float = 0
    min_  : float = float('+inf')
    max_  : float = float('-inf')

    def update(self, new_value : float) -> None:
        """ Uses Welford's algorithm to update variance and trivial procedure to update mean """
        new_value = float(new_value)
        self.count += 1
        delta1 = new_value - self.mean
        self.mean += delta1 / self.count
        delta2 = new_value - self.mean
        self.var = ( delta1 * delta2 + self.var * (self.count - 1) ) / self.count
        self.min_ = min(self.min_, new_value)
        self.max_ = max(self.max_, new_value)

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return (self.count, self.mean, self.var)

def extract_point(raw_val, method) -> float:
    if isinstance(raw_val, list):
        match method:
            case 'mean':
                return float(np.mean(raw_val))
            case 'median':
                return float(np.quantile(raw_val, 0.5, method='closest_observation'))
            case _:
                raise AttributeError("Unknown method passed to extract point")
    elif isinstance(raw_val, RunningMeanVar):
        match method:
            case 'mean':
                return raw_val.mean
            case 'median':
                raise AttributeError("RunningMeanVar cannot track median of collection")
            case _:
                raise AttributeError("Unknown method passed to extract point")
    else:
        raise AttributeError("Unknown type passed to extract point")

def extract_range(raw_val, method) -> tuple[float, float]:
    if isinstance(raw_val, list):
        match method:
            case 'std':
                std = float(np.std(raw_val))
                mean = float(np.mean(raw_val))
                return (mean - std, mean + std)
            case 'Q1-Q3':
                q1q3 = np.quantile(raw_val, [0.25, 0.75], method='closest_observation').tolist()
                return q1q3
            case 'min-max':
                minmax = np.quantile(raw_val, [0.0, 1.0], method='closest_observation').tolist()
                return minmax
            case _:
                raise AttributeError("Unknown method passed to extract point")
    elif isinstance(raw_val, RunningMeanVar):
        match method:
            case 'std':
                std = float(np.sqrt(raw_val.var))
                return (raw_val.mean - std, raw_val.mean + std)
            case 'Q1-Q3':
                raise AttributeError("RunningMeanVar cannot track quantiles of collection")
            case 'min-max':
                return (raw_val.min_, raw_val.max_)
            case _:
                raise AttributeError("Unknown method passed to extract point")
    else:
        raise AttributeError("Unknown type passed to extract point")

_RANGE_NAMES = {
    'std'     : ('-σ', '+σ'),
    'Q1-Q3'   : ('Q1', 'Q3'),
    'min-max' : ('min', 'max')
}
def parse_range_name(name) -> tuple[str, str]:
    if name in _RANGE_NAMES:
        return _RANGE_NAMES[name]
    raise AttributeError(f"Unknown range name: '{name}'")

@dataclass
class RunningValue:
    count : int = 0
    value : Any = None
