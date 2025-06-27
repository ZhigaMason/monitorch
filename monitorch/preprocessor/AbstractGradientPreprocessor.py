
from abc import abstractmethod
from .AbstractPreprocessor import AbstractPreprocessor


class AbstractGradientPreprocessor(AbstractPreprocessor):

    @abstractmethod
    def process_grad(self, name, grad):
        """ Processes gradient  """
        pass

