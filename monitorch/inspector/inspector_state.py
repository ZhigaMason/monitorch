from typing import Callable

class InspectorState:
    """
    TODO
    """

    def __init__(self, is_active_fn : int|Callable[[int], bool] = 1):
        if isinstance(is_active_fn, int):
            cycle_val = is_active_fn
            self.is_active_fn : Callable[[int], bool] = lambda n: n % cycle_val == 0
        else:
            self.is_active_fn = is_active_fn
        self.counter : int = 0
        self.attached = False

    def tick(self, n_ticks : int = 1) -> int:
        """
        Increments :attr:`counter` by n_ticks

        Parameters
        ----------
        n_ticks : int
            Number of ticks to increment by

        Returns
        -------
        The value of :attr:`counter`
        """
        self.counter += n_ticks
        return self.counter

    @property
    def is_active(self) -> bool:
        """
        Activation state according to :attr:`is_active_fn`

        Returns
        -------
        Value of `is_active_fn(counter)`
        """
        return self.is_active_fn(self.counter)
