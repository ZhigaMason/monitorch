"""
    Class to be passed to `nn.Module.register_full_backward_hook`.
    Implements __call__
"""

from monitorch.preprocessor.AbstractBackwardPreprocessor import AbstractBackwardPreprocessor

class BackwardGatherer:

    def __init__(self, preprocessors : list[AbstractBackwardPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name

    def __call__(self, module, args, layer_output) -> None:
        layer_input = args[0]
        for preprocessor in self._preprocessors:
            preprocessor.process(self._name, module, layer_input, layer_output)
