
from abc import ABC, abstractmethod
from typing import Any

class AbstractVizualizer(ABC):

    @abstractmethod
    def vizualize(self, name : str, value : dict[str, Any]) -> None:
        pass
