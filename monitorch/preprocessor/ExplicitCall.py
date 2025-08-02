from typing import Any

from .abstract.abstract_preprocessor import AbstractPreprocessor
from monitorch.numerical import RunningMeanVar

class ExplicitCall(AbstractPreprocessor):
    """
    Class for accumulating data passed by explicit call.

    Object of ``ExplicitCall`` class are provided by :class:`PyTorchInspector`
    to lenses as a foreign preprocessor. ``ExplicitCall`` implements methods to interact directly with its data.
    Its primary usage is to track loss and other performance metrics for :class:`LossMetrics` lens.

    Parameters
    ----------
    train_loss_str : str
        String to save training loss under.
    non_train_loss_str : str
        String to save development, validation or test loss under.

    Attributes
    ----------
    state : dict[str, Any]
        Aggregated data indexed by their names.
    train_loss_str : str
        String to save training loss under.
    non_train_loss_str : str
        String to save non-training loss under.
    """


    def __init__(self, train_loss_str, non_train_loss_str):
        self.state : dict[str, Any] = {}
        self.train_loss_str = train_loss_str
        self.non_train_loss_str = non_train_loss_str

    def push_memory(self, name : str, value) -> None:
        """
        Appends value to container under name and creates a list if there is none.

        Parameters
        ----------
        name : str
            Name under which the value will be saved.
        value
            The value to be saved.
        """
        self.state.setdefault(name, []).append(value)

    def push_running(self, name : str, value : float) -> None:
        """
        Appends value to container under name and creates a :class:`RunningMeanVar` if there is none.

        Parameters
        ----------
        name : str
            Name under which the value will be saved.
        value
            The value to be saved.
        """
        self.state.setdefault(name, RunningMeanVar()).append(value)

    def push_loss(self, value : float, *, train : bool, running : bool = True):
        """
        A utility function to save loss.

        A shorthand to choose whether loss is running and what name to push it under.

        Parameters
        ----------
        value : float
            Value of loss to be saved.
        train : bool
            Whether loss should be saved under :attr:`train_loss_str` or :attr:`non_train_loss_str`
        running : bool
            Indicates if :meth:`push_running` or :meth:`push_memory` should used.
        """
        name = self.train_loss_str if train else self.non_train_loss_str
        if running:
            self.push_running(name, value)
        else:
            self.push_memory(name, value)

    @property
    def value(self) -> dict[str, Any]:
        return self.state

    def reset(self) -> None:
        self.state = {}
