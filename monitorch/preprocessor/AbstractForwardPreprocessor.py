"""
    Base class for all forward pass preprocessors
"""

from abc import abstractmethod
from .AbstractPreprocessor import AbstractPreprocessor


class AbstractForwardPreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process(self, name : str, layer_input, layer_output):
        """ Process raw inputs and outputs into meaningful informaion """
