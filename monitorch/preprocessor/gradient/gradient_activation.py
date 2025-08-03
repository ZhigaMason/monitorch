from typing import Any

from torch import abs as tabs

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractTensorPreprocessor

from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates

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

    def __init__(self, death : bool, inplace : bool, eps : float = 1e-8):
        self._death = death
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list
        self._eps = eps

    def process_tensor(self, name : str, grad):
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

        new_activation_tensor = tabs(grad) > self._eps
        new_activation_rates = reduce_activation_to_activation_rates(new_activation_tensor, batch=False)

        if self._death:
            activations, death_rates = self._value[name]
            death_rates.append(new_activation_rates.eq(0.0).float().mean().item())
            for act in new_activation_rates:
                activations.append(act.item())
        else:
            activations = self._value[name]
            for act in new_activation_rates:
                activations.append(act.item())

    @property
    def value(self) -> dict[str, Any]:
        """ See base class. """
        return self._value

    def reset(self) -> None:
        """ See base class. """
        self._value = {}
