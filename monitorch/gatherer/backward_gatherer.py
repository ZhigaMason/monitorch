from monitorch.preprocessor import AbstractBackwardPreprocessor
from monitorch.inspector.inspector_state import InspectorState
from .abstract_gatherer import AbstractGatherer

class BackwardGatherer(AbstractGatherer):
    """
    Object responsible for collecting data from `torch.nn.Module.register_full_backward_hook`.

    Registers self to module provided in construction as a backward hook,
    on call hands over data and module's name to preprocessors.

    Parameters
    ----------
    module : torch.nn.Module
        Module to hook onto.
    preprocessors : list[:class:`AbstractBackwardPreprocessor`]
        List of preprocessors to hand over data when PyTorch calls the hook.
    name : str
        Name of module to hand over to preprocessors.
    """

    def __init__(self, module, preprocessors : list[AbstractBackwardPreprocessor], name : str, inspector_state : InspectorState):
        super().__init__(inspector_state)
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.register_full_backward_hook(self)

    def detach(self) -> None:
        """
        See base class.
        """
        super().detach()
        self._handle.remove()

    @AbstractGatherer.requires_active_inspector_state
    def __call__(self, module, grad_inp, grad_out) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.process_bw(self._name, module, grad_inp, grad_out)
