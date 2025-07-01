from collections import OrderedDict as odict
from .AbstractVizualizer import AbstractVizualizer
from enum import Enum

from matplotlib import  pyplot as plt
from matplotlib.figure import Figure

class FigType(Enum):
    NUMERICAL = 0
    PROBABILITY = 0
    RELATIONS = 0

class MatplotlibVizualizer(AbstractVizualizer):

    def __init__(self, **kwargs):
        self._to_plot = {}
        self._figure : Figure|None = None
        self._kwargs = kwargs

    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        _, (values, ranges) = self._to_plot.setdefault(main_tag, (FigType.NUMERICAL, (odict(), odict())))

        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, {})
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)

        if not ranges_dict:
            return
        for tag, numerical_ranges_dict in ranges_dict.items():
            tag_dict = ranges.setdefault(tag, {})
            for (desc1, desc2), (y1, y2) in numerical_ranges_dict.items():
                y1s = tag_dict.setdefault(desc1, [])
                y1s.insert(epoch, y1)
                y2s = tag_dict.setdefault(desc2, [])
                y2s.insert(epoch, y2)


    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        _, values = self._to_plot.setdefault(main_tag, (FigType.PROBABILITY, odict()))
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, {})
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)


    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        _, values = self._to_plot.setdefault(main_tag, (FigType.PROBABILITY, odict()))
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, {})
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)

    def show_fig(self) -> Figure:
        if self._figure is None:
            self._figure = self._compose_figure()
        return self._figure


    def _compose_figure(self) -> Figure:
        fig = plt.figure(**self._kwargs)
        return fig
