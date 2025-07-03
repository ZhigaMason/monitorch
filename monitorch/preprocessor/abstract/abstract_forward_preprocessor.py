"""
    Base class for all forward pass preprocessors
"""

from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractForwardPreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process_fw(self, name : str, module, layer_input, layer_output):
        """ Process raw inputs and outputs into meaningful informaion """
        pass
