from .AbstractVisualizer import AbstractVisualizer, TagAttributes
from collections import OrderedDict as odict

class PrintVisualizer(AbstractVisualizer):
    """
    Prints all information to stdout.

    Does not require preconfiguration, nor has any state.
    """

    def __init__(self):
        pass

    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        """
        Prints ``f"{main_tag}: {tag_attr}"``.

        For argument description see base class.
        """
        print(f"{main_tag}: {tag_attr}")

    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        """
        Prints ``f'({epoch}) {main_tag}: {tag} - values : {values_dict.get(tag, {})}; ranges : {ranges_dict.get(tag, {}) if ranges_dict else {}}'``
        for all tags that are in ``values_dict`` or ``ranges_dict``.

        For argument description see base class.
        """
        tags = set(values_dict.keys()) | set(ranges_dict.keys() if ranges_dict else [])
        for tag in tags:
            print(f'({epoch}) {main_tag}: {tag} - values : {values_dict.get(tag, {})}; ranges : {ranges_dict.get(tag, {}) if ranges_dict else {}}')

    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : dict[str, dict[str, float]]) -> None:
        """
        Prints ``f'({epoch}) {main_tag}: {tag} - prbs: {prbs}'``
        for all tags that are in ``values_dict``.

        For argument description see base class.
        """
        for tag, prbs in values_dict.items():
            print(f'({epoch}) {main_tag}: {tag} - prbs: {prbs}')

    def plot_relations(self, epoch : int, main_tag, values_dict : dict[str, dict[str, float]]) -> None:
        """
        Prints ``f'({epoch}) {main_tag}: {tag} - {relations}'``
        for all tags that are in ``values_dict``.

        For argument description see base class.
        """
        for tag, relations in values_dict.items():
            print(f'({epoch}) {main_tag}: {tag} - {relations}')
