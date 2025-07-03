
from monitorch.preprocessor import AbstractGradientPreprocessor
from .AbstractGatherer import AbstractGatherer

class WeightGradientGatherer(AbstractGatherer):

    def __init__(self, module, preprocessors : list[AbstractGradientPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.weight.register_post_accumulate_grad_hook(self)

    def __call__(self, weight):
        for preprocessor in self._preprocessors:
            preprocessor.process_grad(self._name, weight.grad)

    def detach(self) -> None:
        self._handle.remove()
