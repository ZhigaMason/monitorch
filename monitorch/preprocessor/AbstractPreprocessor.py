"""
    Base class for all preprocessors
"""

from abc import ABC, abstractmethod
from typing import Any

from torch.nn import Module

class AbstractPreprocessor(ABC):

    @property
    @abstractmethod
    def value(self) -> dict[str, Any]:
        """ Value computed by Abstract Preprocessor for all layers, that it is processing, identified by name"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        pass
