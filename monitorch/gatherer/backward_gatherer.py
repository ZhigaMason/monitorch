"""
    Class to be passed to `nn.Module.register_full_backward_hook`.
    Implements __call__
"""

from monitorch.preprocessor import AbstractBackwardPreprocessor
from .abstract_gatherer import AbstractGatherer

class BackwardGatherer(AbstractGatherer):

    def __init__(self, module, preprocessors : list[AbstractBackwardPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.register_full_backward_hook(self)

    def detach(self) -> None:
        self._handle.remove()

    def __call__(self, module, grad_inp, grad_out) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.process_bw(self._name, module, grad_inp, grad_out)
