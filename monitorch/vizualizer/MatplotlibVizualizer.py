import numpy as np

from collections import OrderedDict as odict
from enum import Enum
from itertools import chain

from .AbstractVizualizer import AbstractVizualizer, TagAttributes, TagType

from matplotlib import  pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure, SubFigure

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

    _GOLDEN_RATIO = 1.618 # plot w/h ratio

    _RANGE_COLORS = {
        ('min', 'max') : 'grey',
        ('Q1', 'Q3')   : 'blue',
        ('+σ', '-σ')   : 'steelblue'
    }

    _RANGE_ALPHA = 0.2

    _LINE_COLORS = {
        'median' : 'blue',
        'mean'   : 'steelblue',

        'activation_rate' : 'blue',
        'death_rate'      : 'orange',

        'train_loss' : 'blue',
        'val_loss'   : 'orange',

        'train_accuracy'   : 'steelblue',
        'val_accuracy'     : 'red'
    }

    _RELATION_COLORS = [
        'lightblue', 'steelblue'
    ]

    def __init__(self, alt_color=(0.9, 0.9, 0.95), **kwargs):
        self._to_plot = odict()
        self._small_tag_attr : odict[str, TagAttributes] = odict()
        self._big_tag_attr : odict[str, TagAttributes] = odict()
        self._n_max_small_plots : int = -1
        self._figure : Figure|None = None
        self._alt_color = alt_color
        self._kwargs = kwargs

    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        if tag_attr.big_plot:
            self._big_tag_attr[main_tag] = tag_attr
        else:
            self._small_tag_attr[main_tag] = tag_attr


    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        values, ranges = self._to_plot.setdefault(main_tag, (odict(), odict()))

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
                y1s, y2s = tag_dict.setdefault((desc1, desc2), ([], []))
                y1s.insert(epoch, y1)
                y2s.insert(epoch, y2)


    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        values = self._to_plot.setdefault(main_tag, odict())
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, {})
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)


    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        values = self._to_plot.setdefault(main_tag, odict())
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
        self._validate_tag_dictionaries()
        if 'figsize' not in self._kwargs:
            self._kwargs['figsize'] = self._compute_figsize()
        fig = plt.figure(**self._kwargs)
        subfig_dict = self._allocate_subfigures(fig)
        self._plot_tags(subfig_dict)
        return fig

    def _validate_tag_dictionaries(self) -> None:
        raise NotImplemented()

    def _compute_figsize(self) -> tuple[int, int]:
        n_small_tags = len(self._small_tag_attr)
        n_big_tags = len(self._big_tag_attr)

        if self._big_tag_attr == -1:
            self._compute_n_max_small_plots()

        width = 2 * int(max(
            2 * MatplotlibVizualizer._GOLDEN_RATIO * n_big_tags,
            MatplotlibVizualizer._GOLDEN_RATIO * n_small_tags
        ))
        height = 2 * (3 + self._n_max_plots_in_small_tags)

        return (width, height)

    def _compute_n_max_small_plots(self):
        n_max_plots_in_prob_rel = max(
            len(self._to_plot[tag]) for (tag, attr) in self._small_tag_attr if attr in {TagType.PROBABILITY, TagType.RELATIONS}
        )
        numerical_tags = [tag for (tag, attr) in self._small_tag_attr if attr == TagType.NUMERICAL]
        n_max_plots_in_numerical = 0
        for tag in numerical_tags:
            val_dict, range_dict = self._to_plot[tag]
            n_max_plots_in_numerical = max(n_max_plots_in_numerical, len(val_dict), len(range_dict))

        self._n_max_plots_in_small_tags = max(n_max_plots_in_numerical, n_max_plots_in_prob_rel)

    def _allocate_subfigures(self, fig : Figure) -> dict[str, SubFigure]:
        if self._big_tag_attr == -1:
            self._compute_n_max_small_plots()

        height_ratios = (2, self._n_max_plots_in_small_tags + 1)
        gs = GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)

        ret = {}

        up_fig = fig.add_subfigure(gs[0])
        subfigs = up_fig.subfigures(ncols=len(self._big_tag_attr))
        for fig, tag in zip(subfigs, self._big_tag_attr):
            ret[tag] = fig

        lo_fig = fig.add_subfigure(gs[1])
        subfigs = lo_fig.subfigures(ncols=len(self._small_tag_attr))
        for idx, tag in enumerate(self._small_tag_attr):
            subfigs[idx] = (1,1,1) if idx % 2 else self._alt_color
            ret[tag] = subfigs[idx]

        return ret

    def _plot_tags(self, subfig_dict : dict[str, SubFigure]):
        for tag, fig in subfig_dict.items():
            if tag in self._small_tag_attr:
                self._plot_small_tags(fig)
            elif tag in self._big_tag_attr:
                ax = fig.subplots()
                if self._big_tag_attr[tag].logy:
                    ax.set_yscale('log', base=10)
                match self._big_tag_attr[tag].type:
                    case TagType.NUMERICAL:
                        val_dict, range_dict = self._to_plot[tag]
                        MatplotlibVizualizer._plot_numerical(ax, val_dict, range_dict)
                    case TagType.PROBABILITY:
                        MatplotlibVizualizer._plot_probability(ax, self._to_plot[tag])
                    case TagType.RELATIONS:
                        MatplotlibVizualizer._plot_relations(ax, self._to_plot[tag])

    def _plot_small_tags(self, fig : SubFigure) -> None:
        pass

    @staticmethod
    def _plot_numerical(ax, val_dict, range_dict) -> None:
        for range_name, (lo, up) in range_dict.items():
            assert len(lo) == len(up)
            if range_name in MatplotlibVizualizer._RANGE_COLORS:
                ax.fill_between(
                    range(len(lo)), lo, up,
                    color = MatplotlibVizualizer._RANGE_COLORS[range_name],
                    alpha = MatplotlibVizualizer._RANGE_ALPHA
                )
            else:
                ax.fill_between(
                    np.ones(len(lo)), lo, up,
                    alpha = MatplotlibVizualizer._RANGE_ALPHA
                )

        for val_name, values in val_dict:
            if val_name in MatplotlibVizualizer._LINE_COLORS:
                ax.plot(
                    range(len(values)), values,
                    color = MatplotlibVizualizer._LINE_COLORS[val_name]
                )
            else:
                ax.plot(
                    range(len(values)), values,
                )

    @staticmethod
    def _plot_probability(ax, prob_dict) -> None:
        for prob_name, probs in prob_dict.items():
            if prob_name in MatplotlibVizualizer._LINE_COLORS:
                ax.fill_between(
                    range(len(probs)), probs, np.zeros_like(probs),
                    color = MatplotlibVizualizer._LINE_COLORS[prob_name],
                    alpha = MatplotlibVizualizer._RANGE_ALPHA
                )
                ax.plot(
                    range(len(probs)), probs,
                    color = MatplotlibVizualizer._LINE_COLORS[prob_name],
                )
            else:
                ax.fill_between(
                    range(len(probs)), probs, np.zeros_like(probs),
                    alpha = MatplotlibVizualizer._RANGE_ALPHA
                )
                ax.plot(
                    range(len(probs)), probs,
                )

    @staticmethod
    def _plot_relations(ax, rel_dict) -> None:
        #for relations in rel_dict.values():
            # TODO


    @staticmethod
    def range_color(idx):
        if idx < len(MatplotlibVizualizer._RANGE_COLORS):
            return MatplotlibVizualizer._RANGE_COLORS[idx]
        return tuple(np.ones(3) - 1/idx)

    @staticmethod
    def line_color(idx):
        if idx < len(MatplotlibVizualizer._LINE_COLORS):
            return MatplotlibVizualizer._LINE_COLORS[idx]
        return tuple(0.5 * np.ones(3) + 1/idx)


