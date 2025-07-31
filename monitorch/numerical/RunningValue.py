"""
A file containing utility classes used to record running values
"""

import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class RunningMeanVar:
    """
    An object used to keep track of running statistics inplace.

    Collects number of elements, mean, uncorected variance, mininimum and maximum
    of collection through :meth:`update` or :meth:`append` calls.
    """

    count : int   = 0
    """
    Number of update calls on the object. Default is 0.
    """

    mean  : float = 0
    """ Mean calculated through all previous calls. Default is 0. """

    var   : float = 0
    """
    Uncorrected variance (i.e. df = 0) calculated
    from update calls using Welford's algorithm. Default is 0.
    """

    min_  : float = float('+inf')
    """
    Minimal value from update calls. Default is float('+inf')
    """

    max_  : float = float('-inf')
    """
    Maximal value from update calls. Default is float('-inf')
    """

    def update(self, new_value : float) -> None:
        """
        Updates running statistics with provided value.

        Uses Welford's algorithm to update variance and trivial procedure to update mean, minimum and maximum

        Parameters
        ----------
        new_value : float
            The value to update statistics with.
        """
        new_value = float(new_value)
        self.count += 1
        delta1 = new_value - self.mean
        self.mean += delta1 / self.count
        delta2 = new_value - self.mean
        self.var = ( delta1 * delta2 + self.var * (self.count - 1) ) / self.count
        self.min_ = min(self.min_, new_value)
        self.max_ = max(self.max_, new_value)

    append = update
    """
    Alias for :meth:`update` method for compatability with list methods.
    """

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return (self.count, self.mean, self.var)

def extract_point(raw_val : RunningMeanVar|list[float], method : str) -> float:
    """
    Extracts a single variable from :class:`RunningMeanVar` or ``list``.

    Generic function to work with :class:`RunningMeanVar` and lists of floats.

    Parameters
    ----------
    raw_val : :class:`RunningMeanVar` | list[float]
        Object from which the value must be extracted.
    method : str = {'mean', 'median', 'max', 'min', 'Q1', 'Q2', 'std', 'IQR'}
        Description of value to extract.

    Returns
    -------
    float
        Extracted value specified by `method`.

    Raises
    ------
    AttributeError
        If unknown `method` was passed.
        If `method` is **median**, **Q1**, **Q2** or **IQR** and `raw_val` is :class:`RunningMeanVar`
    """
    if isinstance(raw_val, list):
        match method:
            case 'mean':
                return float(np.mean(raw_val))
            case 'median':
                return float(np.quantile(raw_val, 0.5, method='closest_observation'))
            case 'max':
                return float(np.max(raw_val))
            case 'min':
                return float(np.max(raw_val))
            case 'Q1':
                return float(np.quantile(raw_val, 0.25))
            case 'Q3':
                return float(np.quantile(raw_val, 0.25))
            case 'IQR':
                [q1, q3] = np.quantile(raw_val, [0.25, 0.75], method='closest_observation').tolist()
                return q3 - q1
            case 'std':
                return float(np.std(raw_val))
            case _:
                raise AttributeError("Unknown method passed to extract point")
    elif isinstance(raw_val, RunningMeanVar):
        match method:
            case 'mean':
                return raw_val.mean
            case 'max':
                return raw_val.max_
            case 'min':
                return raw_val.min_
            case 'Q1':
                raise AttributeError("RunningMeanVar cannot track 1st quantile of collection")
            case 'Q3':
                raise AttributeError("RunningMeanVar cannot track 3rd quantile of collection")
            case 'median':
                raise AttributeError("RunningMeanVar cannot track median of collection")
            case 'IQR':
                raise AttributeError("RunningMeanVar cannot track IQR of collection")
            case 'std':
                return float(np.sqrt(raw_val.var))
            case _:
                raise AttributeError("Unknown method passed to extract point")
    else:
        raise AttributeError("Unknown type passed to extract point")

def extract_range(raw_val, method) -> tuple[float, float]:
    """
    Extracts a range described by `method` from provided object.

    Generic function to extract ranges (pairs of values) from :class:`RunningMeanVar` or list.

    Parameters
    ----------
    raw_val : :class:`RunningMeanVar` | list[float]
        Object from which the range must be extracted.
    method : str = {'std', 'Q1-Q3', 'min-max'}
        Description of range to extract.

    Returns
    -------
    tuple(float, float)
        Extracted range specified by `method`.

    Raises
    ------
    AttributeError
        If unknown `method` was passed.
        If `method` is **Q1-Q3** and `raw_val` is :class:`RunningMeanVar`
    """
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
    """
    Parses string name into matplotlib annotatable pair of strings.

    Translates::

        'std'     to ('-σ', '+σ')
        'Q1-Q3'   to ('Q1', 'Q3')
        'min-max' to ('min', 'max')

    Parameters
    ----------
    name : str
            Range name

    Returns
    -------
    tuple(str, str)
        Edge names of range

    Raises
    ------
    AttributeError
        If the range name is unknown
    """
    if name in _RANGE_NAMES:
        return _RANGE_NAMES[name]
    raise AttributeError(f"Unknown range name: '{name}'")

@dataclass
class RunningValue:
    count : int = 0
    value : Any = None
