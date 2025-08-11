import numpy as np

from typing_extensions import Self
from collections import OrderedDict as odict
from warnings import warn

from .AbstractVisualizer import AbstractVisualizer, TagAttributes, TagType

from matplotlib import  pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure, SubFigure

class MatplotlibVisualizer(AbstractVisualizer):
    """
    Visualises data using matplotlib.

    Saves data provided by public plot methods, allocates figures, axes and draws plots on :meth:`show_fig`.
    Autoconfigures figures and legends.
    Allocates one superfigure, containing a figure for big plots (big-plot-figure) and a figure for collections of small figures (small-plot-figure).
    For every bigplot tag allcoates another subfigure inside big-plot-figure. Similarly allocates a figure for each small tag inside small-plot-figure.

    Parameters
    ----------
    **kwargs
        Passes all arguments to ``matplotlib.pyplot.figure``.

    Attributes
    ----------
    figure_ : matplotlib.figure.Figure
        Superfigure of the plot.
    """

    _GOLDEN_RATIO = 1.618 # plot w/h ratio

    _UNIT = 2 # figsize multiplier

    _BIG_PLOT_AMP = 2 # big plot size amplifier (small plot height : big plot height = _BIG_PLOT_AMP : 1)

    _SMALL_TITLE_HEIGHT = 0.5 # height of title in small-plot-figure

    _LEGEND_HEIGHT = 1 # height of legend in small-plot-figure

    _LEGEND_ANCHOR = (0.5, 1.2) # anchor of small plot legend

    # colors for chosen ranges
    _RANGE_COLORS = {
        ('min', 'max') : 'grey',
        ('Q1', 'Q3')   : 'blue',
        ('-σ', '+σ')   : 'steelblue',

        ('train_loss min', 'train_loss max')  : 'grey',
        ('train_loss Q1',  'train_loss Q3')   : 'blue',
        ('train_loss -σ',  'train_loss +σ')   : 'steelblue',

        ('val_loss min', 'val_loss max')  : 'bisque',
        ('val_loss Q1',  'val_loss Q3')   : 'orange',
        ('val_loss -σ',  'val_loss +σ')   : 'sandybrown',
    }

    # alpha for ranges
    _RANGE_ALPHA = 0.2

    # colors for chosen lines
    _LINE_COLORS = {
        'median' : 'blue',
        'mean'   : 'steelblue',

        'activation_rate' : 'blue',
        'death_rate'      : 'orange',

        'worst activation_rate' : 'blue',
        'worst death_rate'      : 'orange',

        'train_loss' : 'blue',
        'val_loss'   : 'orange',

        'train_accuracy'   : 'steelblue',
        'val_accuracy'     : 'red'
    }

    # colors to cycle through in relation plot
    _RELATION_COLORS = [
        'cornflowerblue', 'royalblue'
    ]

    # font-weight of figure titles
    _SUPTITLE_WEIGHT = 580

    # colors for small plot faces to cycle through
    _SMALL_TAG_FACE_COLORS = [
        (1,1,1), (0.95, 0.92, 0.9)
    ]

    # warning for no small plots
    _NO_SMALL_TAGS_WARNING = "No small plots, but lenses plotting per layer values used"

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._small_tag_attr = odict()
        self._big_tag_attr = odict()
        self.reset_fig()

    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        """ See base class. """
        if tag_attr.big_plot:
            self._big_tag_attr[main_tag] = tag_attr
        else:
            self._small_tag_attr[main_tag] = tag_attr


    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        """
        Does not draw any plots. Provided data is saved.

        For description of arguments see base class.
        """

        values_ranges = self._to_plot.setdefault(main_tag, odict())

        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values_ranges.setdefault(tag, ({}, {}))[0]
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)

        if not ranges_dict:
            return
        for tag, numerical_ranges_dict in ranges_dict.items():
            tag_dict = values_ranges.setdefault(tag, ({}, {}))[1]
            for (desc1, desc2), (y1, y2) in numerical_ranges_dict.items():
                y1s, y2s = tag_dict.setdefault((desc1, desc2), ([], []))
                y1s.insert(epoch, y1)
                y2s.insert(epoch, y2)


    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        """
        Does not draw any plots. Provided data is saved.

        For description of arguments see base class.
        """
        values = self._to_plot.setdefault(main_tag, odict())
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, {})
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)


    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, odict[str, float]]) -> None:
        """
        Does not draw any plots. Provided data is saved.

        For description of arguments see base class.
        """
        values = self._to_plot.setdefault(main_tag, odict())
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, odict())
            for descriptor, y in numerical_values_dict.items():
                ys = tag_dict.setdefault(descriptor, [])
                ys.insert(epoch, y)

    def show_fig(self) -> Figure:
        """
        Composes figure if it was not shown before and displays it.

        If :attr:`figure_` is not ``None`` returns it.
        Otherwise allocates figures, draws plots defined with :meth:`register_tags` and data provided by plot methods.

        Returns
        -------
        matplotlib.figure.Figure
            Superfigure containing all plots.
        """
        if self.figure_ is None:
            self.figure_ = self._compose_figure()
        return self.figure_

    def reset_fig(self) -> Self:
        """
        Resets figure.

        Sets :attr:`figure_` to ``None`` and internal state to default.

        Returns
        -------
        Self
        """
        self._to_plot = odict()
        self._n_max_small_plots = -1
        self.figure_ = None
        self._n_max_plots_in_small_tags = -1
        return self

    def _compose_figure(self) -> Figure:
        """
        Computes figsize, allocates super and subfigures, draws plots.

        Uses :meth:`_compute_figsize`, :meth:`_allocate_subfigure` and :meth:`_plot_tags`.
        """
        if 'figsize' not in self._kwargs:
            self._kwargs['figsize'] = self._compute_figsize()

        fig = plt.figure(**self._kwargs)

        subfig_dict = self._allocate_subfigures(fig)

        self._plot_tags(subfig_dict)

        return fig

    def _compute_figsize(self) -> tuple[float, float]:
        """
        Computes figure size based on provided tags.

        Width is a maximum of 2x# of big plots and # of small plots times golden ratio.
        Height is maximal number of small plots in small tag + legend and title height + 2 if big plot is present.
        """

        # computes maximal number of small plots if there is no
        if self._n_max_plots_in_small_tags == -1:
            self._compute_n_max_small_plots()
        n_small_tags = len(self._small_tag_attr)
        n_big_tags = len(self._big_tag_attr)

        width = MatplotlibVisualizer._UNIT * int(max(
            MatplotlibVisualizer._BIG_PLOT_AMP * MatplotlibVisualizer._GOLDEN_RATIO * n_big_tags,
            MatplotlibVisualizer._GOLDEN_RATIO * n_small_tags
        ))
        height = MatplotlibVisualizer._UNIT * (MatplotlibVisualizer._SMALL_TITLE_HEIGHT + MatplotlibVisualizer._LEGEND_HEIGHT + self._n_max_plots_in_small_tags)

        return (width, height)

    def _compute_n_max_small_plots(self):
        """
        Computes maximal number of small plots.
        """
        # computes probabalistic and relational as thir subtags occur only in one dictionary.
        n_max_plots_in_prob_rel = max(
            (len(self._to_plot[tag]) for (tag, attr) in self._small_tag_attr.items() if attr.type in {TagType.PROBABILITY, TagType.RELATIONS}),
            default=0
        )

        # parses numerical tags' ranges and values to extract subtags
        numerical_tags = [tag for (tag, attr) in self._small_tag_attr.items() if attr.type == TagType.NUMERICAL]
        n_max_plots_in_numerical = 0
        for tag in numerical_tags:
            val_range_dict = self._to_plot[tag]
            n_max_plots_in_numerical = max(n_max_plots_in_numerical, len(val_range_dict))

        # maximum of two of the previous values
        self._n_max_plots_in_small_tags = max(n_max_plots_in_numerical, n_max_plots_in_prob_rel)

    def _allocate_subfigures(self, fig : Figure) -> dict[str, SubFigure]:
        """
        Allocates subfigures for all tags provided by plot methods.

        Returns
        -------
        dict[str, SubFigure]
            Dictionary of tags and corresponding subfigures.
        """
        assert (len(self._small_tag_attr) + len(self._big_tag_attr)) > 0, "Nothing to plot add lenses or reconfigure them"
        if self._n_max_plots_in_small_tags == -1:
            self._compute_n_max_small_plots()

        # allocates big-plot-figure and small-plot-figure one over another
        # height_ratios are tuned to be such that big-plot is twice taller than small-plot
        height_ratios = (MatplotlibVisualizer._BIG_PLOT_AMP, self._n_max_plots_in_small_tags + MatplotlibVisualizer._LEGEND_HEIGHT + MatplotlibVisualizer._SMALL_TITLE_HEIGHT)
        gs = GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)

        ret = {}

        # allocates big-plot-figure if needed
        # if there is no small plots, allocates the whole superfigure for big-plot-figure
        up_fig : SubFigure
        if len(self._small_tag_attr) == 0:
            up_fig = fig.add_subfigure(GridSpec(1,1)[0])
        elif len(self._big_tag_attr) > 0:
            up_fig = fig.add_subfigure(gs[0])

        # allocates subfigures for individual big-plots
        if len(self._big_tag_attr) > 0:
            subfigs = up_fig.subfigures(ncols=len(self._big_tag_attr), squeeze=False).flatten()
            for subfig, tag in zip(subfigs, self._big_tag_attr):
                ret[tag] = subfig

        # allocates small-plot-figure if needed
        # if there is no big plots, allocates the whole superfigure for small-plot-figure
        lo_fig : SubFigure
        if len(self._big_tag_attr) == 0:
            lo_fig = fig.add_subfigure(GridSpec(1,1)[0])
        elif len(self._small_tag_attr) > 0:
            lo_fig = fig.add_subfigure(gs[1])

        # allocates individual subfigures for small tags
        if len(self._small_tag_attr) > 0:
            subfigs = lo_fig.subfigures(ncols=len(self._small_tag_attr), squeeze=False).flatten()
            for subfig, tag in zip(subfigs, self._small_tag_attr):
                ret[tag] = subfig

        return ret

    def _plot_tags(self, subfig_dict : dict[str, SubFigure]):
        """
        Plots tags onto subfigures provided in ``subfig_dict``.

        Parameters
        ----------
        subfig_dict : dict[str, matplotlib.figure.SubFigure]
            Dictionary with subfigures indexed by tag names.
        """

        # iterates through all tags and plots them
        # saves small figures into a list
        small_figs = []
        for tag, subfig in subfig_dict.items():
            if tag in self._small_tag_attr:
                self._plot_small_tag(subfig, tag)
                small_figs.append(subfig)
            elif tag in self._big_tag_attr:
                subfig.suptitle(tag, fontweight=MatplotlibVisualizer._SUPTITLE_WEIGHT)
                ax = subfig.subplots()
                values = self._to_plot[tag][tag]
                if self._big_tag_attr[tag].logy:
                    ax.set_yscale('log', base=10)
                match self._big_tag_attr[tag].type:
                    case TagType.NUMERICAL:
                        MatplotlibVisualizer._plot_numerical(ax, values[0], values[1])
                    case TagType.PROBABILITY:
                        MatplotlibVisualizer._plot_probability(ax, values)
                    case TagType.RELATIONS:
                        MatplotlibVisualizer._plot_relations(ax, values)
                if self._big_tag_attr[tag].annotate:
                    ax.legend()
                if self._big_tag_attr[tag].ylim is not None:
                    bottom, top = self._big_tag_attr[tag].ylim
                    ax.set_ylim(bottom, top)

        # sets small figures face colors
        colors = MatplotlibVisualizer._SMALL_TAG_FACE_COLORS
        for idx, fig in enumerate(small_figs):
            fig.set_facecolor(colors[idx % len(colors)])

    def _plot_small_tag(self, fig : SubFigure, tag) -> None:
        """
        Iterates through all subtag in tag's dictionary entry.

        Parameters
        ----------
        fig : matplotlib.figure.SubFigure
            Subfigure which the plots will be drawn onto.
        tag : str
            Tag to draw.
        """
        if self._n_max_plots_in_small_tags == 0:
            warn(MatplotlibVisualizer._NO_SMALL_TAGS_WARNING)
            return
        tag_dict = self._to_plot[tag]
        tag_attr = self._small_tag_attr[tag]
        axes = fig.subplots(nrows=self._n_max_plots_in_small_tags, sharex=True, squeeze=False).flatten()
        n_real_plots = len(tag_dict)
        TOTAL_HEIGHT = MatplotlibVisualizer._SMALL_TITLE_HEIGHT + MatplotlibVisualizer._LEGEND_HEIGHT + self._n_max_plots_in_small_tags
        fig.suptitle(
            tag, fontweight=MatplotlibVisualizer._SUPTITLE_WEIGHT,
            y=1 - MatplotlibVisualizer._SMALL_TITLE_HEIGHT / TOTAL_HEIGHT,
            va='baseline'
        )

        # makes unused axes invisible
        for ax in axes[n_real_plots:]:
            ax.set_visible(False)


        # iterates through axes and tags' data
        for ax, (plot_name, values) in zip(axes, tag_dict.items()):
            ax.set_title(plot_name)
            if tag_attr.logy:
                ax.set_yscale('log', base=10)
            match tag_attr.type:
                    case TagType.NUMERICAL:
                        val_dict, range_dict = values
                        MatplotlibVisualizer._plot_numerical(ax, val_dict, range_dict)
                    case TagType.PROBABILITY:
                        MatplotlibVisualizer._plot_probability(ax, values)
                    case TagType.RELATIONS:
                        MatplotlibVisualizer._plot_relations(ax, values)
            if self._small_tag_attr[tag].ylim is not None:
                top, bottom= self._small_tag_attr[tag].ylim
                ax.set_ylim(bottom, top)

        if tag_attr.annotate:
            axes[0].legend(
                loc='lower center',
                bbox_to_anchor=MatplotlibVisualizer._LEGEND_ANCHOR
            )
        # adds ticks to the last axis
        axes[n_real_plots - 1].tick_params(labelbottom=True)
        # adjusts proportion for consitency accross all sizes
        fig.subplots_adjust(top=1 - (MatplotlibVisualizer._SMALL_TITLE_HEIGHT + MatplotlibVisualizer._LEGEND_HEIGHT) / TOTAL_HEIGHT, bottom=0)

    @staticmethod
    def _plot_numerical(ax, val_dict, range_dict) -> None:
        """
        Function to plot numerical values.

        Parameters
        ----------
        ax : Axis
            Matplotlib axis which will be the plots drawn onto.
        val_dict : dict[str, list[float]]
            Dictionary of values (lines) to plot.
        range_dict : dict[tuple[str, str], tuple[list[float], list[float]]]
            Dictionary of ranges to plot.
        """
        for range_name, (lo, up) in range_dict.items():
            assert len(lo) == len(up)
            if range_name in MatplotlibVisualizer._RANGE_COLORS:
                ax.fill_between(
                    range(len(lo)), lo, up,
                    color = MatplotlibVisualizer._RANGE_COLORS[range_name],
                    alpha = MatplotlibVisualizer._RANGE_ALPHA,
                    label = f"{range_name[0]} -- {range_name[1]}"
                )
            else:
                ax.fill_between(
                    range(len(lo)), lo, up,
                    alpha = MatplotlibVisualizer._RANGE_ALPHA,
                    label = f"{range_name[0]} -- {range_name[1]}"
                )

        for val_name, values in val_dict.items():
            if val_name in MatplotlibVisualizer._LINE_COLORS:
                ax.plot(
                    range(len(values)), values,
                    color = MatplotlibVisualizer._LINE_COLORS[val_name],
                    label=val_name
                )
            else:
                ax.plot(
                    range(len(values)), values,
                    label=val_name
                )

    @staticmethod
    def _plot_probability(ax, prob_dict) -> None:
        """
        Function to plot proportions.

        Parameters
        ----------
        ax : Axis
            Matplotlib axis which will be the plots drawn onto.
        prob_dict : dict[str, list[float]]
            Dictionary of values to plot
        """
        for prob_name, probs in prob_dict.items():
            if prob_name in MatplotlibVisualizer._LINE_COLORS:
                ax.fill_between(
                    range(len(probs)), probs, np.zeros_like(probs),
                    color = MatplotlibVisualizer._LINE_COLORS[prob_name],
                    alpha = MatplotlibVisualizer._RANGE_ALPHA
                )
                ax.plot(
                    range(len(probs)), probs,
                    color = MatplotlibVisualizer._LINE_COLORS[prob_name],
                    label=prob_name
                )
            else:
                ax.fill_between(
                    range(len(probs)), probs, np.zeros_like(probs),
                    alpha = MatplotlibVisualizer._RANGE_ALPHA
                )
                ax.plot(
                    range(len(probs)), probs,
                    label=prob_name
                )
        ax.set_ylim(0, 1)

    @staticmethod
    def _plot_relations(ax, rel_dict) -> None:
        """
        Function to plot relations.

        Parameters
        ----------
        ax : Axis
            Matplotlib axis which will be the plots drawn onto.
        rel_dict : dict[str, list[float]]
            Dictionary of values to plot indexed by subtags.
        """
        if not rel_dict:
            return
        l = len(next(iter(rel_dict.values())))
        first_record = []
        for relations in rel_dict.values():
            assert l == len(relations), "All relations must have same number of epochs recorded"
            first_record.append(relations[0])
        ax.stackplot(range(l), *rel_dict.values(), colors=MatplotlibVisualizer._RELATION_COLORS)
        arr = np.array(first_record)
        pos_arr = np.cumsum(arr) - arr / 2
        for pos, rel_name in zip(pos_arr, rel_dict.keys()):
            ax.text(0, pos, rel_name)
