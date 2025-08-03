
from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates

from torch import no_grad, is_grad_enabled, Tensor, abs as tabs
from typing import Any
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
    record_no_grad : bool
        Indicator if outputs made with ``torch.no_grad`` must be preprocessed.
        It is expected that validation pass is done without gradient computation.
    eps : float
        Numerical constant under which value is regarded as a zero.
    """

    def __init__(self, death : bool, inplace : bool, record_no_grad : bool, eps : float = 1e-8):
        self._death = death
        self._value = {} # Either name -> activation or name -> (activation, death_tensor)
        self._thresholds : dict[str, tuple[float, float]]= {}
        self._agg_class = RunningMeanVar if inplace else list
        self._record_no_grad = record_no_grad
        self._eps = eps

    def process_fw(self, name : str, module, layer_input, layer_output) -> None:
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
        if not (self._record_no_grad or is_grad_enabled()):
            return
        if name not in self._value:
            if self._death:
                self._value[name] = (self._agg_class(), self._agg_class())
            else:
                self._value[name] = self._agg_class()

        new_activation_tensor : Tensor
        new_activation_rate : Tensor
        with no_grad():
            new_activation_tensor = tabs(layer_output) > self._eps
            new_activation_rate = reduce_activation_to_activation_rates(new_activation_tensor, batch=True)

        if self._death:
            activations, death_rates = self._value[name]
            death_rates.append(new_activation_rate.eq(0).float().mean())
            for act in new_activation_rate:
                activations.append(act.item())
        else:
            activations = self._value[name]
            for act in new_activation_rate:
                activations.append(act.item())


    @property
    def value(self) -> dict[str, Any]:
        """ See base class. """
        return self._value

    def reset(self) -> None:
        """ See base class. """
        self._value = {}
