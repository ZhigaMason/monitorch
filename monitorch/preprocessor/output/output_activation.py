from typing import Any

from torch import Tensor, is_grad_enabled, no_grad
from torch import abs as tabs
from torch import float32 as tfloat32

from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor


class OutputActivation(AbstractForwardPreprocessor):
    """
    Preprocessor to record activations of outputs.

    We say that a neuron or a channel is activated if the output is non-zero
    (information is propagated forward). If neuron is not activated for all samples in a batch,
    we say it is dead. Death rate is a proportion of dead neurons against layer size.

    Parameters
    ----------
    death : bool
        Indicator if death rate is to be collected.
    inplace : bool
        Indicator if :class:`RunningMeanVar` or ``list`` should be used for aggregation.
    record_eval : bool
        Indicator if outputs during evaluation must be preprocessed.
    evaluation_from_grad : bool
        Flag indicating if evaluation passes should be considered from gradient or modele.training
    eps : float
        Numerical constant under which value is regarded as a zero.
    channel_last : bool
        If ``True``, expects data in ``[batch, seq_len, ..., features]`` format where the feature/channel
        dimension is last (e.g. transformer outputs). If ``False`` (default), expects PyTorch's standard
        ``[batch, features, spatial_dims, ...]`` format.
    """

    def __init__(self, death: bool, inplace: bool, record_eval: bool, evaluation_from_grad: bool, eps: float = 1e-8, channel_last: bool = False):
        self._death = death
        self._value = {}  # Either name -> activation or name -> (activation, death_tensor)
        self._thresholds: dict[str, tuple[float, float]] = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._record_eval = record_eval
        self._is_train = (lambda m: is_grad_enabled()) if evaluation_from_grad else (lambda m: m.training)
        self._eps = eps
        self._channel_last = channel_last

    def process_fw(self, name: str, module, layer_input, layer_output) -> None:
        """
        Computes activation from layer output.

        Flattens spatial dimensions, computes activations and saves each sample.
        Computes death rate if ``death=True`` was set.

        Parameters
        ----------
        name : str
            Name of the module which outputs are processed.
        module : torch.nn.Module
            Module object, its outputs are processed.
        layer_input : torch.Tensor
            Should be input of layer, but it is ignored in this method.
        layer_output : torch.Tensor
            Outputs to compute activations from.
        """
        if not (self._record_eval or self._is_train(module)):
            return
        if name not in self._value:
            if self._death:
                self._value[name] = (self._agg_class(), self._agg_class())
            else:
                self._value[name] = self._agg_class()

        new_activation_tensor: Tensor
        new_activation_rate: Tensor
        with no_grad():
            new_activation_tensor = tabs(layer_output) > self._eps
            if self._channel_last:
                new_activation_tensor = new_activation_tensor.movedim(-1, 1)
            new_activation_rate = reduce_activation_to_activation_rates(new_activation_tensor, batch=True)

        if self._death:
            activations, death_rates = self._value[name]
            death_rates.append(new_activation_rate.eq(0).mean(dtype=tfloat32))
            activations.append(new_activation_rate.mean(dtype=tfloat32))
        else:
            activations = self._value[name]
            activations.append(new_activation_rate.mean(dtype=tfloat32))

    @property
    def value(self) -> dict[str, Any]:
        """See base class."""
        return self._value

    def reset(self) -> None:
        """See base class."""
        self._value = {}
