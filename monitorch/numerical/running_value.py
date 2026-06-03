"""
A file containing utility classes used to record running values
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class RunningMeanVar:
    """
    An object used to keep track of running statistics inplace.

    Collects number of elements, mean, uncorected variance, mininimum and maximum
    of collection through :meth:`update` or :meth:`append` calls.
    """

    count: int = 0
    """
    Number of update calls on the object. Default is 0.
    """

    mean: float = 0
    """ Mean calculated through all previous calls. Default is 0. """

    var: float = 0
    """
    Uncorrected variance (i.e. df = 0) calculated
    from update calls using Welford's algorithm. Default is 0.
    """

    min_: float = float('+inf')
    """
    Minimal value from update calls. Default is float('+inf')
    """

    max_: float = float('-inf')
    """
    Maximal value from update calls. Default is float('-inf')
    """

    _handle: None = None
    """
    Handle for synchronization of async operations.
    """

    _gathered_data = None
    """
    Data gathered on master rank for visualization.
    """

    def update(self, new_value: float) -> None:
        """
        Updates running statistics with provided value.

        Uses Welford's algorithm to update variance and trivial procedure to update mean, minimum and maximum

        Parameters
        ----------
        new_value : float
            The value to update statistics with.
        """
        if hasattr(new_value, 'detach'):
            new_value = new_value.detach()
        new_value = float(new_value)
        self.count += 1
        delta1 = new_value - self.mean
        self.mean += delta1 / self.count
        delta2 = new_value - self.mean
        self.var = (delta1 * delta2 + self.var * (self.count - 1)) / self.count
        self.min_ = min(self.min_, new_value)
        self.max_ = max(self.max_, new_value)

    append = update
    """
    Alias for :meth:`update` method for compatability with list methods.
    """

    def merge(self, other: 'RunningMeanVar') -> None:
        """
        Uses Chan algorithm to merge another RunningMeanVar.

        Parameters
        ----------
        other : RunningMeanVar
        """

        if other.count == 0:
            return  # Nothing to merge from other

        if self.count == 0:
            # If self is empty, just copy from other
            self.count = other.count
            self.mean = other.mean
            self.var = other.var
            self.min_ = other.min_
            self.max_ = other.max_
            return

        # Both self.count and other.count are > 0, proceed with Chan's algorithm

        m2_self = self.var * self.count
        m2_other = other.var * other.count

        new_count = self.count + other.count
        delta = other.mean - self.mean
        new_mean = (self.count * self.mean + other.count * other.mean) / new_count
        new_m2 = m2_self + m2_other + delta**2 * self.count * other.count / new_count
        new_var = new_m2 / new_count

        new_min_ = min(self.min_, other.min_)
        new_max_ = max(self.max_, other.max_)

        self.count = new_count
        self.mean = new_mean
        self.var = new_var
        self.min_ = new_min_
        self.max_ = new_max_

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return (self.count, self.mean, self.var)

    def start_sync(self, dst_rank: int = 0):
        """
        Synchronizes statistics across all distributed ranks.
        Only the dst_rank will hold the mathematically correct global statistics.

        Parameters
        ----------

        rmv : RunningMeanVar
            The values to scatter

        dst_rank : int = 0
            Rank to gather all of the stats at.
        """
        current_device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

        local_stats = torch.tensor(
            [self.count, self.mean, self.var, self.min_, self.max_],
            dtype=torch.float64,  # float64 prevents precision loss in variance
            device=current_device,
        )

        world_size = dist.get_world_size()
        self._gathered_data = [torch.zeros_like(local_stats) for _ in range(world_size)]

        self._handle = dist.all_gather(
            self._gathered_data,
            local_stats,
            async_op=True,
        )

    def finish_sync(self) -> None:
        assert self._handle is not None, 'There is no started communication.'

        self._handle.wait()
        self._handle = None

        if self._gathered_data is None:
            return

        # Chan's algorithm for distributed variance calculation
        # it can be done as a tree reduction, but realistically the world size < 16
        global_count = 0
        global_mean = 0.0
        global_S = 0.0
        global_min = float('+inf')
        global_max = float('-inf')

        for stat in self._gathered_data:
            N_B = int(stat[0].item())
            if N_B == 0:
                continue

            mean_B = stat[1].item()
            var_B = stat[2].item()
            S_B = var_B * N_B

            if global_count == 0:
                global_count = N_B
                global_mean = mean_B
                global_S = S_B
                global_min = stat[3].item()
                global_max = stat[4].item()
            else:
                N_A = global_count
                mean_A = global_mean

                # Chan's algorithm updates
                delta = mean_B - mean_A
                global_count = N_A + N_B
                global_mean = mean_A + delta * (N_B / global_count)
                global_S = global_S + S_B + (delta**2) * (N_A * N_B / global_count)

                # Trivial min/max updates
                global_min = min(global_min, stat[3].item())
                global_max = max(global_max, stat[4].item())

        # Update the object inplace on Master
        self.count = global_count
        self.mean = global_mean
        self.var = global_S / global_count if global_count > 0 else 0.0
        self.min_ = global_min
        self.max_ = global_max
        self._gathered_data = None


Accumulator = RunningMeanVar | list[float]


def start_sync_rmv_or_error(rmv: Accumulator, dst_rank: int):
    """
    Does nothing in single-process setting.
    Starts synchronization in distributed settings on RMV.

    Otherwise raises error.

    Parameters
    ----------
    rmv : Accumulator
        RunningMeanVarn to sync.
    dst_rank : int
        Rank to gather data at.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if not isinstance(rmv, RunningMeanVar):
        raise AttributeError(f'Cannot sync {rmv.__class__}. Try collecting inplace metrics.')
    rmv.start_sync(dst_rank)


def finish_sync_rmv_or_error(rmv: Accumulator):
    """
    Does nothing in single-process setting.
    Finishes synchronization in distributed settings on RMV.

    Otherwise raises error.

    Parameters
    ----------
    rmv : Accumulator
        RunningMeanVarn to sync.
    dst_rank : int
        Rank to gather data at.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if not isinstance(rmv, RunningMeanVar):
        raise AttributeError(f'Cannot sync {rmv.__class__}. Try collecting inplace metrics.')
    rmv.finish_sync()


def extract_point(raw_val: Accumulator, method: str) -> float:
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
                raise AttributeError('Unknown method passed to extract point')
    elif isinstance(raw_val, RunningMeanVar):
        match method:
            case 'mean':
                return raw_val.mean
            case 'max':
                return raw_val.max_
            case 'min':
                return raw_val.min_
            case 'Q1':
                raise AttributeError('RunningMeanVar cannot track 1st quantile of collection')
            case 'Q3':
                raise AttributeError('RunningMeanVar cannot track 3rd quantile of collection')
            case 'median':
                raise AttributeError('RunningMeanVar cannot track median of collection')
            case 'IQR':
                raise AttributeError('RunningMeanVar cannot track IQR of collection')
            case 'std':
                return float(np.sqrt(raw_val.var))
            case _:
                raise AttributeError('Unknown method passed to extract point')
    else:
        raise AttributeError('Unknown type passed to extract point')


def extract_range(raw_val: Accumulator, method) -> tuple[float, float]:
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
                raise AttributeError('Unknown method passed to extract point')
    elif isinstance(raw_val, RunningMeanVar):
        match method:
            case 'std':
                std = float(np.sqrt(raw_val.var))
                return (raw_val.mean - std, raw_val.mean + std)
            case 'Q1-Q3':
                raise AttributeError('RunningMeanVar cannot track quantiles of collection')
            case 'min-max':
                return (raw_val.min_, raw_val.max_)
            case _:
                raise AttributeError('Unknown method passed to extract point')
    else:
        raise AttributeError('Unknown type passed to extract point')


_RANGE_NAMES = {'std': ('-σ', '+σ'), 'Q1-Q3': ('Q1', 'Q3'), 'min-max': ('min', 'max')}


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
    count: int = 0
    value: Any = None
