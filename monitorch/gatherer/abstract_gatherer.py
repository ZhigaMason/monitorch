from functools import wraps
from abc import ABC, abstractmethod

class AbstractGatherer(ABC):
    """
    An abstract class that parents all gatherers.
    """

    def __init__(self, inspector_state):
        self.inspector_state = inspector_state

    @abstractmethod
    def detach(self) -> None:
        """
        Abstract method to detach from module.

        Detaches gatherer and all its acompaning preprocessors from module.
        """
        self.inspector_state = None

    @staticmethod
    def requires_active_inspector_state(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            #
            # signature wrapper(self, *args, **kwargs) does not behave well with unpacking, namely it starts to scaffold it like this (arg1, (arg2, (arg3, ...)))
            #
            instance = args[0]
            if instance.inspector_state.is_active:
                fn(*args, **kwargs)
        return wrapper
