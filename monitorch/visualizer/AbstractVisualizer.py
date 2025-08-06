from abc import ABC, abstractmethod
from collections import OrderedDict as odict
from dataclasses import dataclass
from enum import Enum


class TagType(Enum):
    """
    Enum of types of plots. Used in :class:`TagAttributes`.
    """

    NUMERICAL = 0
    """
    Numerical line and band-range plot, no implicit y-axis limits.
    """
    PROBABILITY = 1
    """
    Line plot with implicit y-axis limits (0, 1).
    """
    RELATIONS = 2
    """
    Displays comparison between several numerical variables.
    Plot method depends on visualizer: stackplot for :class:`MatplotlibVisualizer`, multiline plot for :class:`TensorBoardVisualizer`.
    """

@dataclass
class TagAttributes:
    """
    Packaged data about plot.

    Used by visualizers, must be given by lens using :meth:`register_tags`.
    """

    logy : bool
    """
    logy : bool
        Indicator if scale of y-axis must be log, flag is relecant only for :class:`MatplotlibVisualizer`.
    """

    big_plot : bool
    """
    bigplot : bool
        Indicator if plot should be big or a collection of small ones, flag is relecant only for :class:`MatplotlibVisualizer`.
    """

    annotate : bool
    """
    annotate : bool
        Indicator if legend must be plotted, flag is relecant only for :class:`MatplotlibVisualizer`.
    """

    type : TagType
    """
    type : :class:`TagType`
        Type of plot.
    """

    ylim : tuple[float, float]|None = None
    """
    ylim : tuple[flaot, float]|None
        Optional limits for y-axis.
    """

    def __repr__(self) -> str:
        return f"TagAttributes(logy={self.logy}, big_plot={self.big_plot}, annotate={self.annotate}, type={self.type.name}, ylim={self.ylim})"


class AbstractVisualizer(ABC):
    """
    Base class for all visualizers.

    :class:`PyTorchInspector` and lenses from :mod:`monitorch.lens` use methods provided by this class for vizualization.
    """

    @abstractmethod
    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        """
        Prepare visualizer's inner state for plot.

        Parameters
        ----------
        main_tag : str
            Name of the collection of plots.
        tag_attr : TagAttributes
            Tag attributes to configure plot.
        """
        pass

    @abstractmethod
    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        """
        Plots numerical values and ranges for a collection of tags.

        For single big plot use ``main_tag`` as an only subtag in ``values_dict`` and ``ranges_dict``.
        Plots single epoch values, does not process a sequence of points.

        Parameters
        ----------
        epoch : int
            Training epoch.
        main_tag : str
            Name of the collection of plots.
        values_dict : OrderedDict[str, dict[str,float]]
            Ordered dictionary of subtags and their numerical values.
            Most common is an ordered dict of layer names and layer statstics.

            Example
            -------
            ::

                OrderedDict([
                    ('lin1', {'mean': 0.0,  'median': 1.0}),
                    ('lin2', {'mean': 42.0, 'median': 10.0})
                ])

            There is no strict restriction for each subtag to have the same dictionary structure,
            though some visualizers may not behave as expected if inner dictionaries differ.
        ranges_dict : OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]|None
            Optional ordered dictionary of subtags and its ranges.
            Most common is an ordered dict of layer names and layer statstics range.

            Example
            -------
            ::

                OrderedDict([
                    ('lin1', {'min': 1.0,  'max': 10.0}),
                    ('lin2', {'min': -4.0, 'max': 0.0})
                ])

            There is no strict restriction for each subtag to have the same dictionary structure,
            though some visualizers may not behave as expected if inner dictionaries differ.
        """
        pass

    @abstractmethod
    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        """
        Plots proportions/probabilities for a collection of tags.

        For single big plot use ``main_tag`` as an only subtag in ``values_dict``.
        Plots single epoch values, does not process a sequence of points.
        Sets y-axis limits to (0, 1) if it is possible to do programmatically.

        Parameters
        ----------
        epoch : int
            Training epoch.
        main_tag : str
            Name of the collection of plots.
        values_dict : OrderedDict[str, dict[str, float]]
            Ordered dictionary of subtags and propotions.
            Most common is a dictionary of layer names and layer activations/death rates.

            Example
            -------
            ::

                OrderedDict([
                    ('relu1', {'activation_rate' : 0.8, death_rate : 0.15}),
                    ('relu2 , {'activation_rate' : 0.6, death_rate : 0.2 })
                ])

            There is no strict restriction for each subtag to have the same dictionary structure,
            though some visualizers may not behave as expected if inner dictionaries differ.
        """
        pass

    @abstractmethod
    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        """
        Plot comparison for a collection of tags.

        For single big plot use ``main_tag`` as an only subtag in ``values_dict``.
        Plots single epoch values, does not process a sequence of points.

        Parameters
        ----------
        epoch : int
            Training epoch.
        main_tag : str
            Name of the collection of plots.
        values_dict : OrderedDict[str, dict[str, float]]
            Ordered dictionary of subtags and statistics.
            Most common is a big plot with layer information.

            Example
            -------
            ::

                OrderedDict([
                    ('Output Log Norm', {
                        'relu1' : 1.2,
                        'relu2' : 1.5
                    })
                ])
        """
        pass
