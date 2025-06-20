"""
    A file containing utility classes used to record running values
"""

from dataclasses import dataclass
from typing import Any

@dataclass
class RunningMeanVar:
    count : int   = 0
    mean  : float = 0
    var   : float = 0

    def update(self, new_value : float) -> None:
        """ Uses Welford's algorithm to update variance and trivial procedure to update mean """
        self.count += 1
        delta1 = new_value - self.mean
        self.mean += delta1 / self.count
        delta2 = new_value - self.mean
        self.var = ( delta1 * delta2 + self.var * (self.count - 1) ) / self.count

    def __iter__(self):
        return (self.count, self.mean, self.var)

@dataclass
class RunningValue:
    count : int = 0
    value : Any = None
