
from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates

from torch import no_grad, isclose, tensor
from typing import Any
from monitorch.preprocessor.abstract.abstract_forward_preprocessor import AbstractForwardPreprocessor
from monitorch.numerical.threshold_for_module import threshold_for_module


class OutputActivation(AbstractForwardPreprocessor):

    def __init__(self, death : bool, inplace : bool):
        self._death = death
        self._value = {} # Either name : activation or name : (activation, death_tensor)
        self._thresholds : dict[str, tuple[float, float]]= {}
        self._agg_class = RunningMeanVar if inplace else list

    @no_grad
    def process_fw(self, name : str, module, layer_input, layer_output) -> None:
        if name not in self._value:
            if self._death:
                self._value[name] = (self._agg_class(), self._agg_class())
            else:
                self._value[name] = self._agg_class()
            self._thresholds[name] = threshold_for_module(module)

        lo, up = self._thresholds[name]

        new_activation_tensor = (isclose(layer_output, tensor([lo])) | isclose(layer_output, tensor([up]))).logical_not()
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
        """ Value computed by Abstract Preprocessor for all layers, that it is processing, identified by name"""
        return self._value

    def reset(self) -> None:
        """ Resets preprocessor for further computation """
        self._value = {}
