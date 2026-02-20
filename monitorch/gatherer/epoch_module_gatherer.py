from monitorch.preprocessor import AbstractModulePreprocessor

from .abstract_gatherer import AbstractGatherer


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

    def __init__(self, module, preprocessors: list[AbstractModulePreprocessor], name: str, inspector_state):
        super().__init__(inspector_state)
        self._module = module
        self._name: str = name
        self._preprocessors = preprocessors

    @AbstractGatherer.requires_active_inspector_state
    def __call__(self):
        for ppr in self._preprocessors:
            ppr.process_module(self._name, self._module)

    def detach(self) -> None:
        """
        See base class
        """
        super().detach()
        self._module = None
        self._name = ''
