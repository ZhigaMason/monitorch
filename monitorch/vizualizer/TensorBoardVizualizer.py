
from .AbstractVizualizer import AbstractVizualizer
from torch.utils.tensorboard import SummaryWriter

class TensorBoardVizualizer(AbstractVizualizer):

    def __init__(self, logdir, **kwargs):
        self.writer = SummaryWriter(logdir, **kwargs)

    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : dict[str, dict[str, float]], ranges_dict : dict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:

        ranges_dict = ranges_dict or {}
        ranges_decomposed : dict[str, dict[str, float]]= {}
        for tag, ranges in ranges_dict.items():
            tmp_dict = {k[0]:v[0] for k,v in ranges.items()} | {k[1]:v[1] for k,v in ranges.items()}
            ranges_decomposed[tag] = tmp_dict

        keys = set(ranges_decomposed.keys()) | set(values_dict.keys())
        to_plot = {}
        for key in keys:
            to_plot[key] = ranges_decomposed.get(key, {}) | values_dict.get(key, {})

        for tag, subtag_scalar in to_plot:
            self.writer.add_scalars(main_tag + '/' + tag, subtag_scalar, global_step=epoch)

    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : dict[str, dict[str, float]]) -> None:
        for tag, prbs in values_dict:
            self.writer.add_scalars(main_tag + '/' + tag, prbs, global_step=epoch)

    def plot_relations(self, epoch : int, main_tag, values_dict : dict[str, dict[str, float]]) -> None:
        for tag, relations in values_dict:
            self.writer.add_scalars(main_tag + '/' + tag, relations, global_step=epoch)
