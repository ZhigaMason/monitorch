
from monitorch.preprocessor import AbstractTensorPreprocessor
from .abstract_gatherer import AbstractGatherer

class ParameterGradientGatherer(AbstractGatherer):
    """
    Class to collect gradients from attributes of module.

    Object of ``ParameterGradientGatherer`` gatherer is a stateful callback registered
    onto ``torch.Tensor`` using ``register_post_accumulate_grad_hook``. On call hands over data to preprocessors.

    Parameters
    ----------
    parameter : str
        Name of learnable parameter in module to gather data from.
    module : torch.nn.Module
        Module from which the learnable parameter is obtained. The data will be collected from that learnable parameter.
    preprocessors : list[:class:`AbstractTensorPreprocessor`]
        Preprocessors that will aggregate data.
    name : str
        Name of the module.
    """

    def __init__(self, parameter : str, module, preprocessors : list[AbstractTensorPreprocessor], name : str):
        self._preprocessors = preprocessors
        self._name = name
        self._handle = getattr(module, parameter).register_post_accumulate_grad_hook(self)

    def __call__(self, parameter):
        for preprocessor in self._preprocessors:
            preprocessor.process_tensor(self._name, parameter.grad)

    def detach(self) -> None:
        """
        See base class
        """
        self._handle.remove()
