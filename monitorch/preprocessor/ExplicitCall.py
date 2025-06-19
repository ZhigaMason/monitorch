"""
    Base class for all preprocessors
"""

from typing import Any
from torch.nn import Module

from .AbstractPreprocessor import AbstractPreprocessor
from monitorch.numerical import RunningMeanVar

class ExplicitCall(AbstractPreprocessor):

    def __init__(self):
        self.state : dict[str, Any] = {}

    def push_remember(self, name : str, value) -> None:
        """ pushes value into list """
        if name not in self.state:
            self.state[name] = []
        self.state[name].append(value)

    def push_mean_var(self, name : str, value : float) -> None:
        """ Updates running mean and variance in state """
        if name not in self.state:
            self.state[name] = RunningMeanVar()
        self.state[name].update(value)

    def value(self) -> dict[str, Any]:
        """ Value is state accumulated in inspector calls """
        return self.state

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self.state = {}

    def is_preprocessing(self, _ : Module) -> bool:
        """ Does not serve any module, but explicit function calls on inspector """
        return False
