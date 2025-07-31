
from abc import ABC, abstractmethod


class AbstractGatherer(ABC):
    """
    An abstract class that parents all gatherers.
    """

    @abstractmethod
    def detach(self) -> None:
        """
        Abstract method to detach from module.

        Detaches gatherer and all its acompaning preprocessors from module.
        """
