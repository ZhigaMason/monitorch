
from torch import no_grad, is_grad_enabled
from torch.linalg import vector_norm
from math import sqrt

from typing import Any
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor
from monitorch.numerical import RunningMeanVar

class OutputNorm(AbstractForwardPreprocessor):
    """
    Preprocessor to compute norms of outputs.

    Flattens spatial and channel/neuron dimensions of output,
    computes L2 norm or RMS (if normalized) of flattened vectors and takes mean over a batch.

    Parameters
    ----------
    normalize : bool
        Indicator if output norm should be normalized by square root of number of elements in single sample output.
    inplace : bool
        Indicator if :class:`RunningMeanVar` or ``list`` should be used for aggregation.
    record_no_grad : bool
        Indicator if outputs made with ``torch.no_grad`` must be preprocessed.
        It is expected that validation pass is done without gradient computation.
    """

    def __init__(self, normalize : bool, inplace : bool, record_no_grad : bool):
        self._normalize = normalize
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._agg_class = RunningMeanVar if inplace else list
        self._record_no_grad = record_no_grad

    def process_fw(self, name : str, module, layer_input, layer_output):
        """
        Computes mean output norm.

        Flattens spatial and channel dimensions, computes (normalized) norm of individual samples
        and saves their average.

        Parameters
        ----------
        name : str
            Name of the module which outputs are processed.
        module : torch.nn.Module
            Module object, its outputs are processed.
        layer_input : torch.Tensor
            Should be input of layer, but it is ignored in this method.
        layer_output : torch.Tensor
            Outputs to compute norm from.
        """
        if not (self._record_no_grad or is_grad_enabled()):
            return
        norm_container = self._value.setdefault(name, self._agg_class())

        with no_grad():
            norm_mean = vector_norm(layer_output.flatten(1, -1), dim=-1).mean().item()
            if self._normalize:
                norm_mean /= sqrt(layer_output[0].numel())
            norm_container.append(norm_mean)

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
