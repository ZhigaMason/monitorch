from monitorch.preprocessor import AbstractForwardPreprocessor

from .abstract_gatherer import AbstractGatherer


class FeedForwardGatherer(AbstractGatherer):
    """
    Object responsible for collecting data from `torch.nn.Module.register_forward_hook`.

    Registers self to module provided in construction as a forward hook,
    on call hands over data and module's name to preprocessors.

    Parameters
    ----------
    module : torch.nn.Module
        Module to hook onto.
    preprocessors : list[:class:`AbstractForwardPreprocessor`]
        List of preprocessors to hand over data when PyTorch calls the hook.
    name : str
        Name of module to hand over to preprocessors.
    """

    def __init__(self, module, preprocessors: list[AbstractForwardPreprocessor], name: str, inspector_state):
        super().__init__(inspector_state)
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.register_forward_hook(self)

    def detach(self) -> None:
        """
        See base class
        """
        super().detach()
        self._handle.remove()

    @AbstractGatherer.requires_active_inspector_state
    def __call__(self, module, args, layer_output) -> None:
        layer_input = args[0]
        for preprocessor in self._preprocessors:
            preprocessor.process_fw(self._name, module, layer_input, layer_output)
