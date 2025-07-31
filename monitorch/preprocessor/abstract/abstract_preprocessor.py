
from abc import ABC, abstractmethod
from typing import Any

class AbstractPreprocessor(ABC):
    """
    Base class for all preprocessors.
    """

    @property
    @abstractmethod
    def value(self) -> dict[str, Any]:
        """
        Value computed by preprocessor for all layers, that it is processing, identified by name.

        Returns
        -------
        dict[str, Any]
            Result of computations done from creation of preprocessor or last reset.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        pass
