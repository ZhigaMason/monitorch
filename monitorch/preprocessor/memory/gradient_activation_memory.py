from typing import Any

from torch import isclose, tensor

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractGradientPreprocessor

from monitorch.numerical import reduce_activation_to_activation_rates

class GradientActivationMemory(AbstractGradientPreprocessor):

    def __init__(self, death : bool, eps : float = 1e-7):
        self._death = death
        self._value = {}
        self._eps = eps

    def process_grad(self, name : str, grad):
        if name not in self._value:
            if self._death:
                self._value[name] = ([], [])
            else:
                self._value[name] = []

        new_activation_tensor = isclose(grad, tensor(0.0)).logical_not()
        new_activation_rates = reduce_activation_to_activation_rates(new_activation_tensor, batch=False)
        if self._death:
            activations, death_rates = self._value[name]
            activations.extend(new_activation_rates.tolist())
            death_rates.append(new_activation_rates.eq(0.0).float().mean().item())
        else:
            self._value[name].extend(new_activation_rates.tolist())



    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
