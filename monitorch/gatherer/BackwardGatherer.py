"""
    Class to be passed to `nn.Module.register_full_backward_hook`.
    Implements __call__
"""

from monitorch.preprocessor.AbstractBackwardPreprocessor import AbstractBackwardPreprocessor

class BackwardGatherer:

    def __init__(self, preprocessors : list[AbstractBackwardPreprocessor], name : str):
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
            preprocessor.process(self._name, layer_input, layer_output)
