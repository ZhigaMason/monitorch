"""
    Class to be passed to `nn.Module.register_forward_hook`.
    Implements __call__
"""

from monitorch.preprocessor.AbstractForwardPreprocessor import AbstractForwardPreprocessor

class FeedForwardGatherer:

    def __init__(self, preprocessors : list[AbstractForwardPreprocessor], name):
        self._preprocessors = preprocessors
        self._name = name
        self.handle = None

    def detach(self) -> bool:
        if self.handle is None:
            return False
        self.handle.remove()
        return True

    def __call__(self, module, args, layer_output) -> None:
        layer_input = args[0]
        for preprocessor in self._preprocessors:
            preprocessor.process(self._name, module, layer_input, layer_output)
