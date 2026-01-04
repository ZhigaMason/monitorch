from .abstract_gatherer import AbstractGatherer
from monitorch.preprocessor import AbstractTensorPreprocessor
from torch.optim import Optimizer

class OptimizerStepParameterGatherer(AbstractGatherer):
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

    def __init__(self, module, optimizer : Optimizer, parameter : str, preprocessors : list[AbstractTensorPreprocessor], name : str, inspector_state):
        super().__init__(inspector_state)
        self._module = module
        self._parameter : str = parameter
        self._name : str = name
        self._preprocessors = preprocessors
        self._handle = optimizer.register_step_post_hook(self)

    @AbstractGatherer.requires_active_inspector_state
    def __call__(self, *args, **kwargs):
        for ppr in self._preprocessors:
            ppr.process_tensor(self._name, getattr(self._module, self._parameter))

    def detach(self) -> None:
        """
        See base class
        """
        super().detach()
        self._module = None
        self._name = ''
        self._handle.remove()
