
from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractModulePreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process_module(self, name : str, module):
        """ extracts information from module object """
        pass
