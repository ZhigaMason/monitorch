import os
import re
from collections import OrderedDict as odict
from .AbstractVisualizer import AbstractVisualizer, TagAttributes
from torch.utils.tensorboard import SummaryWriter

class TensorBoardVisualizer(AbstractVisualizer):

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
            general_tag = TensorBoardVisualizer.transform_tag_str(main_tag, tag)
            for subtag, value in tag_dict.items():
                plot_tag = TensorBoardVisualizer.transform_tag_str(general_tag, subtag)
                self.writer.add_scalar(plot_tag, value, global_step=epoch)

        if ranges_dict:
            for tag, tag_dict in ranges_dict.items():
                general_tag = TensorBoardVisualizer.transform_tag_str(main_tag, tag)
                for (subtag1, subtag2), (value1, value2) in tag_dict.items():
                    plot_tag = TensorBoardVisualizer.transform_tag_str(general_tag, subtag1)
                    self.writer.add_scalar(plot_tag, value1, global_step=epoch)
                    plot_tag = TensorBoardVisualizer.transform_tag_str(general_tag, subtag2)
                    self.writer.add_scalar(plot_tag, value2, global_step=epoch)


    def plot_probabilities(self, epoch : int, main_tag : str, values_dict : odict[str, dict[str, float]]) -> None:
        for tag, prbs_dict in values_dict.items():
            general_tag = TensorBoardVisualizer.transform_tag_str(main_tag, tag)
            for subtag, prb in prbs_dict.items():
                plot_tag = TensorBoardVisualizer.transform_tag_str(general_tag, subtag)
                self.writer.add_scalar(plot_tag, prb, global_step=epoch)

    def plot_relations(self, epoch : int, main_tag, values_dict : odict[str, dict[str, float]]) -> None:
        for tag, relations in values_dict.items():
            plot_tag = TensorBoardVisualizer.transform_tag_str(main_tag, tag)
            self.writer.add_scalars(plot_tag, relations, global_step=epoch)

    @staticmethod
    def transform_tag_str(general_tag, subtag, delimiter = '/') -> str:
        return re.sub(
            'σ', 'std',
            re.sub(' ', '_',
                   (general_tag + delimiter + subtag).lower()
            )
        )

