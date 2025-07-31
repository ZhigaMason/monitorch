from .abstract_gatherer import AbstractGatherer
from monitorch.preprocessor import AbstractModulePreprocessor

class EpochModuleGatherer(AbstractGatherer):
    """
    Gatherer to hand over whole module object on call.

    Keeps a reference of module to pass it on call to preprocessors with name attached.

    Parameters
    ----------
    module : torch.nn.Module
        Module to be handed over to preprocessors.
    preprocessors : list[:class:`AbstractModulePreprocessor`]
        Preprocessors to hand the module over to.
    name : str
        Name of the module.
    """

    def __init__(self, module, preprocessors : list[AbstractModulePreprocessor], name : str):
        self._module = module
        self._name : str = name
        self._preprocessors = preprocessors

    def __call__(self):
        for ppr in self._preprocessors:
            ppr.process_module(self._name, self._module)

    def detach(self) -> None:
        """
        See base class
        """
        self._module = None
        self._name = ''
