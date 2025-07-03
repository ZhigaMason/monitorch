from torch import no_grad, ones_like, bool as bool_
from typing import Any
from .AbstractForwardPreprocessor import AbstractForwardPreprocessor
from ._module_classes import isactivation, threshold_for_module
from monitorch.numerical import reduce_activation_to_activation_rates


class OutputActivationMemory(AbstractForwardPreprocessor):

    def __init__(self, death : bool, eps : float = 1e-7):
        self._death = death
        self._value = {} # Either name : list[activation] or name : (list[activation], list[death_rates])
        self._thresholds : dict[str, tuple[float, float]]= {}
        self._eps : float = eps

    @no_grad
    def process(self, name : str, module, layer_input, layer_output) -> None:
        if name not in self._value:
            if self._death:
                self._value[name] = ([], [])
            else:
                self._value[name] = []
            self._thresholds[name] = threshold_for_module(module)

        lo, up = self._thresholds[name]

        new_activation_tensor = ((layer_output - lo).abs() > self._eps) & ((layer_output - up).abs() > self._eps)
        new_activation_rate = reduce_activation_to_activation_rates(new_activation_tensor, batch=True)

        if self._death:
            activations, death_rates = self._value[name]
            activations.extend(new_activation_rate.tolist())
            death_rates.append(new_activation_rate.eq(0).float().mean().item())
        else:
            self._value[name].extend(new_activation_rate.tolist())


    @property
    def value(self) -> dict[str, Any]:
        """ Value computed by Abstract Preprocessor for all layers, that it is processing, identified by name"""
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
