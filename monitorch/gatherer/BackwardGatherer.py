"""
    Class to be passed to `nn.Module.register_full_backward_hook`.
    Implements __call__
"""

from monitorch.preprocessor.AbstractBackwardPreprocessor import AbstractBackwardPreprocessor
from .AbstractGatherer import AbstractGatherer

class BackwardGatherer(AbstractGatherer):

    def __init__(self, module, preprocessors : list[AbstractBackwardPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.register_forward_hook(self)

    def detach(self) -> None:
        self._handle.remove()

    def __call__(self, module, args, layer_output) -> None:
        layer_input = args[0]
        for preprocessor in self._preprocessors:
            preprocessor.process(self._name, module, layer_input, layer_output)
