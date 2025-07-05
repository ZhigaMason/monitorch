from typing import Any

from torch import isclose, tensor

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractGradientPreprocessor

from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates

class GradientActivation(AbstractGradientPreprocessor):

    def __init__(self, death : bool, inplace : bool):
        self._death = death
        self._value = {}
        self._agg_class = RunningMeanVar if inplace else list

    def process_grad(self, name : str, grad):
        if name not in self._value:
            if self._death:
                self._value[name] = (self._agg_class(), self._agg_class())
            else:
                self._value[name] = self._agg_class()

        new_activation_tensor = isclose(grad, tensor(0.0)).logical_not()
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
        return self._value

    def reset(self) -> None:
        self._value = {}
