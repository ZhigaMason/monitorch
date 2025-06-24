from abc import ABC, abstractmethod

class AbstractVizualizer(ABC):

    @abstractmethod
    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : dict[str, dict[str, float]], ranges_dict : dict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        pass

    @abstractmethod
    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : dict[str, dict[str, float]]) -> None:
        pass

    @abstractmethod
    def plot_relations(self, epoch : int, main_tag, values_dict : dict[str, dict[str, float]]) -> None:
        pass
