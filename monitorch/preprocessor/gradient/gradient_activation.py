from typing import Any

from torch import abs as tabs
from torch import float32 as tfloat32
from torch import no_grad

from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates, start_sync_rmv_or_error, finish_sync_rmv_or_error
from monitorch.preprocessor.abstract.abstract_tensor_preprocessor import AbstractTensorPreprocessor


class GradientActivation(AbstractTensorPreprocessor):
    """
    Preprocessor class to compute gradient activaitions and death.

    We define a neuron to be active if it has non-zero gradient at any datapoint in a batch iteration,
    it is dead otherwise. This preprocessor calcualtes death rate and activations over an epoch.
    Death rate is a proportion of dead neurons in each batch.
    It can be further aggregated into mean or median accross all batch iterations in an epoch.

    Parameters
    ----------
    death : bool
        Flag indicating if death rate should be computed.
    inplace : bool
        Flag indicating whether to collect data inplace using :class:`RunningMeanVar` or to stack them into a list.
    eps : float
        Numerical constant under which value is regarded as a zero.
    """

    def __init__(self, death: bool, inplace: bool, eps: float = 1e-8):
        self._death = death
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._eps = eps

    def process_tensor(self, name: str, grad):
        """
        Computes activation and death rate on a gradient.

        Transforms gradient into a boolean mask, applies :func:`reduce_activation_to_activation_rates`.
        Activation rates are saved and used to compute death rate.

        Parameters
        ----------
        name : str
            Name of a source of gradient.
        grad : torch.Tensor
            Gradient tensor to compute activations from.
        """
        if name not in self._value:
            if self._death:
                self._value[name] = (self._agg_class(), self._agg_class())
            else:
                self._value[name] = self._agg_class()

        with no_grad():
            new_activation_tensor = tabs(grad) > self._eps
            new_activation_rates = reduce_activation_to_activation_rates(new_activation_tensor, batch=False)

        if self._death:
            activations, death_rates = self._value[name]
            death_rates.append(new_activation_rates.eq(0.0).mean(dtype=tfloat32))
            activations.append(new_activation_rates.mean(dtype=tfloat32))
        else:
            activations = self._value[name]
            activations.append(new_activation_rates.mean(dtype=tfloat32))

    @property
    def value(self) -> dict[str, Any]:
        """See base class."""
        return self._value

    def start_sync(self, dst_rank: int = 0) -> None:
        """
        Syncs the data with the dst_rank.

        Parameters
        ----------
        dst_rank : int = 0
            Master rank to gather data at.
        """
        if self._death:
            for act, death in self._value.values():
                start_sync_rmv_or_error(act, dst_rank=dst_rank)
                start_sync_rmv_or_error(death, dst_rank=dst_rank)
        else:
            for act in self._value.values():
                start_sync_rmv_or_error(act, dst_rank=dst_rank)

    def finish_sync(self, dst_rank: int = 0) -> None:
        """
        Finishes syncing the data with the dst_rank.
        """
        if self._death:
            for act, death in self._value.values():
                finish_sync_rmv_or_error(act)
                finish_sync_rmv_or_error(death)
        else:
            for act in self._value.values():
                finish_sync_rmv_or_error(act)

    def reset(self) -> None:
        """See base class."""
        self._value = {}
