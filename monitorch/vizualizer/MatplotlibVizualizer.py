from collections import OrderedDict as odict
from enum import Enum
from itertools import chain

from .AbstractVizualizer import AbstractVizualizer

from matplotlib import  pyplot as plt
from matplotlib.figure import Figure, SubFigure

class FigType(Enum):
    NUMERICAL = 0
    PROBABILITY = 1
    RELATIONS = 2

# this is such a mess
# should be rewritten with separate ordered dictionaries for each plot method
#
# should add more degrees of separation most probably
# current layout cannot deal with one lens providing data for distinct parameters (e.g. weight and bias)
# it will collide in plot numerical
#
# concept of "main_tag" should be more clear.
# is it 1 main_tag per lens or is it 1 main_tag per displayed statistics
# second makes it easier to think allocate subfigures as they all will have width 1
# allocating subplots will also be easier
#
# TODO: redefine vizualizer arch, this is garbage
#
# the whole thing should be rethought
#
# allocating subfigures and subplots is a@@

class MatplotlibVizualizer(AbstractVizualizer):

    def __init__(self, **kwargs):
        self._to_plot = odict()
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
        subfig_dict = self._allocate_subfigures(fig)
        return fig

    def _allocate_subfigures(self, fig : Figure) -> dict[str, SubFigure]:
        expected_sizes = self._compute_expected_sizes_dict()
        return {}

    def _compute_expected_sizes_dict(self) -> dict[str, tuple[int, int]]:
        ret = {}
        for main_tag, (fig_type, tag_dict) in self._to_plot.items():
            h, w = 0, 0
            match fig_type:
                case FigType.NUMERICAL:
                    val_tag_dict, range_tag_dict = tag_dict # super confusing: tag_dict is actually a 2-tuple

                    subtages = set()
                case FigType.PROBABILITY | FigType.RELATIONS:
                    h = len(tag_dict)
                    w = max(
                        int(len(val) > 0) for val in tag_dict.values()
                    )
            ret[main_tag] = (h, w)

        return ret

