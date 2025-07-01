from abc import ABC, abstractmethod
from collections import OrderedDict as odict

class AbstractVizualizer(ABC):

    @abstractmethod
    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        pass

    @abstractmethod
    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        pass

    @abstractmethod
    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        pass
