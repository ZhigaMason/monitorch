from collections import OrderedDict as odict

from .AbstractVisualizer import AbstractVisualizer, TagAttributes


class DummyVisualizer(AbstractVisualizer):
    """
    Visualizer that does nothing. Created for easier interaction in distributed setting.
    """

    def register_tags(self, main_tag: str, tag_attr: TagAttributes) -> None:
        """
        See base class.
        """
        pass

    def plot_numerical_values(
        self,
        epoch: int,
        main_tag: str,
        values_dict: odict[str, dict[str, float]],
        ranges_dict: odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None,
    ) -> None:
        """
        See base class.
        """
        pass

    def plot_probabilities(
        self,
        epoch: int,
        main_tag: str,
        values_dict: odict[str, dict[str, float]],
    ) -> None:
        """
        See base class.
        """
        pass

    def plot_relations(self, epoch: int, main_tag, values_dict: odict[str, dict[str, float]]) -> None:
        """
        See base class.
        """
        pass
