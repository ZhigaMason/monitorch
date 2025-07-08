from collections import OrderedDict as odict
from .AbstractVizualizer import AbstractVizualizer, TagAttributes
from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardVizualizer(AbstractVizualizer):

    def __init__(self, log_dir = None, comment = '', **kwargs):
        if not log_dir:
            # stolen directly from SummaryWriter implementation
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time + "_" + socket.gethostname() + comment
            )
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment, **kwargs)

    def register_tags(self, main_tag : str, tag_attr : TagAttributes) -> None:
        """ Tensorboard needs no registration. Is present for consitency. """
        pass

    def plot_numerical_values(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]], ranges_dict : odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:

        for tag, tag_dict in values_dict.items():
            general_tag = main_tag + '/' + tag
            for subtag, value in tag_dict.items():
                self.writer.add_scalar(general_tag + '/' + subtag, value, global_step=epoch)

        if ranges_dict:
            for tag, tag_dict in ranges_dict.items():
                general_tag = main_tag + '/' + tag
                for (subtag1, subtag2), (value1, value2) in tag_dict.items():
                    self.writer.add_scalar(general_tag + '/' + subtag1, value1, global_step=epoch)
                    self.writer.add_scalar(general_tag + '/' + subtag2, value2, global_step=epoch)


    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        for tag, prbs_dict in values_dict.items():
            general_tag = main_tag + '/' + tag
            for sub_tag, prb in prbs_dict.items():
                self.writer.add_scalar(general_tag + '/' + sub_tag, prb, global_step=epoch)

    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        for tag, relations in values_dict.items():
            self.writer.add_scalars(main_tag + '/' + tag, relations, global_step=epoch)
