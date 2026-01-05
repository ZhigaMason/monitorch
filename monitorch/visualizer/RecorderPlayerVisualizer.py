import pickle
from collections import OrderedDict as odict

from .AbstractVisualizer import AbstractVisualizer, TagAttributes

class RecorderVisualizer(AbstractVisualizer):
    """
    Serializes all visualizer calls to a binary file using pickle.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Open file in write-binary mode with buffering
        self.file = open(filepath, 'wb')

    def __del__(self):
        self.close()

    def _record(self, method_name: str, **kwargs):
        """Helper to dump method name and arguments to file."""
        payload = {
            'method': method_name,
            'kwargs': kwargs
        }
        pickle.dump(payload, self.file)

    def register_tags(self, main_tag: str, tag_attr: TagAttributes) -> None:
        self._record('register_tags', main_tag=main_tag, tag_attr=tag_attr)

    def plot_numerical_values(self, epoch: int, main_tag: str,
                              values_dict: odict[str, dict[str, float]],
                              ranges_dict: odict[str, dict[tuple[str, str], tuple[float, float]]] | None = None) -> None:
        self._record('plot_numerical_values',
                     epoch=epoch,
                     main_tag=main_tag,
                     values_dict=values_dict,
                     ranges_dict=ranges_dict)

    def plot_probabilities(self, epoch: int, main_tag: str, values_dict: odict[str, dict[str, float]]) -> None:
        self._record('plot_probabilities', epoch=epoch, main_tag=main_tag, values_dict=values_dict)

    def plot_relations(self, epoch: int, main_tag, values_dict: odict[str, dict[str, float]]) -> None:
        self._record('plot_relations', epoch=epoch, main_tag=main_tag, values_dict=values_dict)

    def close(self):
        """Safely close the file handle."""
        if hasattr(self, 'file') and self.file and not self.file.closed:
            self.file.flush()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PlayerVisualizer:
    """
    Reads a recorded visualizer file and executes calls on a target instance.
    """
    def __init__(self, source_filepath: str, target_visualizer: AbstractVisualizer):
        self.source_filepath = source_filepath
        self.target = target_visualizer

    def playback(self):
        """
        Iterates through the pickle file and triggers methods on the target visualizer.
        """
        try:
            with open(self.source_filepath, 'rb') as f:
                while True:
                    try:
                        record = pickle.load(f)
                        method_name = record['method']
                        kwargs = record['kwargs']

                        if not hasattr(self.target, method_name):
                            print(f"Warning: Target visualizer missing method '{method_name}'")
                            continue
                        method = getattr(self.target, method_name)
                        method(**kwargs)

                    except EOFError:
                        break
        except FileNotFoundError:
            print(f"Error: File {self.source_filepath} not found.")
