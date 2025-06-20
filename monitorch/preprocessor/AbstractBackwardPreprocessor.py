"""
    Base class for all backward pass preprocessors
"""

from abc import abstractmethod
from .AbstractPreprocessor import AbstractPreprocessor


class AbstractBackwardPreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process(self, name : str, module, grad_input, grad_output):
        """ Process raw gradients into meaningful informaion """
