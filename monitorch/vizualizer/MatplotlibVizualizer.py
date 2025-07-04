import numpy as np

from collections import OrderedDict as odict

from .AbstractVizualizer import AbstractVizualizer, TagAttributes, TagType

from matplotlib import  pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure, SubFigure

class MatplotlibVizualizer(AbstractVizualizer):

    _GOLDEN_RATIO = 1.618 # plot w/h ratio

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
        'cornflowerblue', 'royalblue'
    ]

    _SUPTITLE_WEIGHT = 580

    _SMALL_TAG_FACE_COLORS = [
        (1,1,1), (0.95, 0.92, 0.9)
    ]

    def __init__(self, **kwargs):
        self._to_plot = odict()
        self._small_tag_attr : odict[str, TagAttributes] = odict()
        self._big_tag_attr : odict[str, TagAttributes] = odict()
        self._n_max_small_plots : int = -1
        self._figure : Figure|None = None
        self._kwargs = kwargs
        self._n_max_plots_in_small_tags : int = -1

    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        print('AAAAAAAAAAAAAAA')
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


    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, odict[str, float]]) -> None:
        values = self._to_plot.setdefault(main_tag, odict())
        for tag, numerical_values_dict in values_dict.items():
            tag_dict = values.setdefault(tag, odict)
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
        print(fig)
        subfig_dict = self._allocate_subfigures(fig)
        self._plot_tags(subfig_dict)
        return fig

    def _validate_tag_dictionaries(self) -> None:
        # I cant remember what i had in mind
        #                                :D
        pass

    def _compute_figsize(self) -> tuple[int, int]:
        if self._n_max_plots_in_small_tags == -1:
            self._compute_n_max_small_plots()
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
            (len(self._to_plot[tag]) for (tag, attr) in self._small_tag_attr if attr in {TagType.PROBABILITY, TagType.RELATIONS}),
            default=0
        )
        numerical_tags = [tag for (tag, attr) in self._small_tag_attr if attr == TagType.NUMERICAL]
        n_max_plots_in_numerical = 0
        for tag in numerical_tags:
            val_dict, range_dict = self._to_plot[tag]
            n_max_plots_in_numerical = max(n_max_plots_in_numerical, len(val_dict), len(range_dict))

        self._n_max_plots_in_small_tags = max(n_max_plots_in_numerical, n_max_plots_in_prob_rel)

    def _allocate_subfigures(self, fig : Figure) -> dict[str, SubFigure]:
        assert (len(self._small_tag_attr) + len(self._big_tag_attr)) > 0, "Nothing to plot add lenses or reconfigure them"
        print('1', fig)
        if self._n_max_plots_in_small_tags == -1:
            self._compute_n_max_small_plots()

        height_ratios = (2, self._n_max_plots_in_small_tags + 1)
        gs = GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)

        ret = {}

        print('2', fig)
        up_fig : SubFigure
        if len(self._small_tag_attr) == 0:
            up_fig = fig.add_subfigure(GridSpec(1,1)[0])
        elif len(self._big_tag_attr) > 0:
            up_fig = fig.add_subfigure(gs[0])

        print('3', fig)
        if len(self._big_tag_attr) > 0:
            subfigs = up_fig.subfigures(ncols=len(self._big_tag_attr), squeeze=False).flatten()
            for subfig, tag in zip(subfigs, self._big_tag_attr):
                print('OOO', tag, subfig)
                ret[tag] = subfig

        lo_fig : SubFigure
        if len(self._big_tag_attr) == 0:
            lo_fig = fig.add_subfigure(GridSpec(1,1)[0])
        elif len(self._small_tag_attr) > 0:
            lo_fig = fig.add_subfigure(gs[1])

        if len(self._small_tag_attr) > 0:
            subfigs = lo_fig.subfigures(ncols=len(self._small_tag_attr), squeeze=False).flatten()
            for subfig, tag in zip(subfigs, self._small_tag_attr):
                ret[tag] = subfig

        return ret

    def _plot_tags(self, subfig_dict : dict[str, SubFigure]):
        small_figs = []
        for tag, subfig in subfig_dict.items():
            print('I', tag, subfig)
            if tag in self._small_tag_attr:
                self._plot_small_tag(subfig, tag)
                small_figs.append(subfig)
            elif tag in self._big_tag_attr:
                subfig.suptitle(tag, fontweight=MatplotlibVizualizer._SUPTITLE_WEIGHT)
                ax = subfig.subplots()
                if self._big_tag_attr[tag].logy:
                    ax.set_yscale('log', base=10)
                match self._big_tag_attr[tag].type:
                    case TagType.NUMERICAL:
                        print(self._to_plot)
                        val_dict, range_dict = self._to_plot[tag]
                        MatplotlibVizualizer._plot_numerical(ax, val_dict[tag], range_dict[tag])
                    case TagType.PROBABILITY:
                        MatplotlibVizualizer._plot_probability(ax, self._to_plot[tag][tag])
                    case TagType.RELATIONS:
                        MatplotlibVizualizer._plot_relations(ax, self._to_plot[tag][tag])

        colors = MatplotlibVizualizer._SMALL_TAG_FACE_COLORS
        for idx, fig in enumerate(small_figs):
            fig.set_facecolor(colors[idx % len(colors)])

    def _plot_small_tag(self, fig : SubFigure, tag) -> None:
        tag_dict = self._to_plot[tag]
        tag_attr = self._small_tag_attr[tag]
        axes = fig.subplots(nrows=self._n_max_plots_in_small_tags, sharex=True)
        n_real_plots = len(tag_dict)
        fig.suptitle(tag, fontweight=MatplotlibVizualizer._SUPTITLE_WEIGHT)
        for ax in axes[n_real_plots:]:
            ax.set_visible(False)

        for ax, (plot_name, values) in zip(axes, tag_dict.items()):
            ax.set_title(plot_name)
            if tag_attr.logy:
                ax.set_yscale('log', base=10)
            match tag_attr.type:
                    case TagType.NUMERICAL:
                        val_dict, range_dict = values
                        MatplotlibVizualizer._plot_numerical(ax, val_dict, range_dict)
                    case TagType.PROBABILITY:
                        MatplotlibVizualizer._plot_probability(ax, values)
                    case TagType.RELATIONS:
                        MatplotlibVizualizer._plot_relations(ax, values)
        axes[n_real_plots - 1].tick_params(labelbottom=True)
        fig.subplots_adjust(top=0.95 * (1 - 1/(self._n_max_plots_in_small_tags + 1) ** 2),bottom=0)

    @staticmethod
    def _plot_numerical(ax, val_dict, range_dict) -> None:
        print('\n\n\n\n\n\n\n')
        print('range_dict', range_dict)
        print('\nval_dict', val_dict)
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

        for val_name, values in val_dict.items():
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
        l = len(next(iter(rel_dict.values())))
        first_record = []
        for relations in rel_dict.values():
            assert l == len(relations), "All relations must have same number of epochs recorded"
            first_record.append(relations[0])
        ax.stackplot(range(l), *rel_dict.values(), colors=MatplotlibVizualizer._RELATION_COLORS)
        arr = np.array(first_record)
        pos_arr = np.cumsum(arr) - arr / 2
        for pos, rel_name in zip(pos_arr, rel_dict.keys()):
            ax.text(0, pos, rel_name)


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


