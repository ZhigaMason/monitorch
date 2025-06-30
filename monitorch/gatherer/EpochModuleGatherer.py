
from .AbstractGatherer import AbstractGatherer
from monitorch.preprocessor import AbstractModulePreprocessor

class EpochModuleGatherer(AbstractGatherer):

    def __init__(self, module, preprocessors : list[AbstractModulePreprocessor], name : str):
        self._module = module
        self._name : str = name
        self._preprocessors = preprocessors

    def __call__(self):
        for ppr in self._preprocessors:
            ppr.process_module(self._name, self._module)

    def detach(self) -> None:
        self._module = None
        self._name = ''
