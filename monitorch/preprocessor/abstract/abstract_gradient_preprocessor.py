
from abc import abstractmethod
from .abstract_preprocessor import AbstractPreprocessor


class AbstractGradientPreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process_grad(self, name, grad):
        """ Processes gradient  """
        pass

