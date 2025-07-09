
from monitorch.preprocessor import AbstractGradientPreprocessor
from .abstract_gatherer import AbstractGatherer

class ParameterGradientGatherer(AbstractGatherer):

    def __init__(self, parameter : str, module, preprocessors : list[AbstractGradientPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = getattr(module, parameter).register_post_accumulate_grad_hook(self)

    def __call__(self, parameter):
        for preprocessor in self._preprocessors:
            preprocessor.process_grad(self._name, parameter.grad)

    def detach(self) -> None:
        self._handle.remove()
