"""
    Base class for all preprocessors
"""

from typing import Any

from monitorch.preprocessor.abstract.abstract_preprocessor import AbstractPreprocessor
from monitorch.numerical import RunningMeanVar

class ExplicitCall(AbstractPreprocessor):

    def __init__(self, train_loss_str, non_train_loss_str):
        self.state : dict[str, Any] = {}
        self.train_loss_str = train_loss_str
        self.non_train_loss_str = non_train_loss_str

    def push_memory(self, name : str, value) -> None:
        """ pushes value into list """
        self.state.setdefault(name, []).append(value)

    def push_running(self, name : str, value : float) -> None:
        """ Updates running mean and variance in state """
        self.state.setdefault(name, RunningMeanVar()).update(value)

    def push_loss(self, value : float, *, train : bool, running : bool = True):
        name = self.train_loss_str if train else self.non_train_loss_str
        if running:
            self.push_running(name, value)
        else:
            self.push_memory(name, value)

    def value(self) -> dict[str, Any]:
        """ Value is state accumulated in inspector calls """
        return self.state

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self.state = {}
