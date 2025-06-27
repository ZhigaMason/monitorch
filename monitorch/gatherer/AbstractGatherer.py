
from abc import ABC, abstractmethod


class AbstractGatherer(ABC):

    @abstractmethod
    def detach(self) -> None:
        """ detaches gatherer and all its acompaning preprocessors from module """
