from math import sqrt
from typing import Any

from torch import is_grad_enabled, no_grad
from torch.linalg import vector_norm

from monitorch.numerical import RunningMeanVar
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor


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
    record_eval : bool
        Indicator if outputs during evaluation must be preprocessed.
    evaluation_from_grad : bool
        Flag indicating if evaluation passes should be considered from gradient or modele.training
    channel_last : bool
        If ``True``, expects data in ``[batch, seq_len, ..., features]`` format where the feature/channel
        dimension is last (e.g. transformer outputs). If ``False`` (default), expects PyTorch's standard
        ``[batch, features, spatial_dims, ...]`` format. The norm computation is equivalent in both cases
        since all non-batch dimensions are flattened before computing the L2 norm.
    """

    def __init__(self, normalize: bool, inplace: bool, record_eval: bool, evaluation_from_grad: bool, channel_last: bool = False):
        self._normalize = normalize
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._record_eval = record_eval
        self._is_train = (lambda m: is_grad_enabled()) if evaluation_from_grad else (lambda m: m.training)
        self._channel_last = channel_last

    def process_fw(self, name: str, module, layer_input, layer_output):
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
        if not (self._record_eval or self._is_train(module)):
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
