from abc import ABC, abstractmethod
from collections import OrderedDict as odict
from dataclasses import dataclass
from enum import Enum


class TagType(Enum):
    NUMERICAL = 0
    PROBABILITY = 1
    RELATIONS = 2

@dataclass
class TagAttributes:
    logy : bool
    big_plot : bool
    type : TagType

    def __repr__(self) -> str:
        return f"TagAttributes(logy={self.logy}, big_plot={self.big_plot}), type={self.type.name})"


class AbstractVizualizer(ABC):

    @abstractmethod
    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        pass

    @abstractmethod
    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        pass

    @abstractmethod
    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        pass

    @abstractmethod
    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        pass
