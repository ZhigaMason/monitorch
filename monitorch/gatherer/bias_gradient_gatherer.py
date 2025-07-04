
from monitorch.preprocessor import AbstractGradientPreprocessor
from .abstract_gatherer import AbstractGatherer

class BiasGradientGatherer(AbstractGatherer):

    def __init__(self, module, preprocessors : list[AbstractGradientPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = module.bias.register_post_accumulate_grad_hook(self)

    def __call__(self, bias):
        for preprocessor in self._preprocessors:
            preprocessor.process_grad(self._name, bias.grad)

    def detach(self) -> None:
        self._handle.remove()
