from typing import Any

from torch import isclose, tensor

from monitorch.preprocessor.abstract.abstract_gradient_preprocessor import AbstractGradientPreprocessor

from monitorch.numerical import RunningMeanVar, reduce_activation_to_activation_rates

class GradientActivationRunning(AbstractGradientPreprocessor):

    def __init__(self, death : bool, eps : float = 1e-7):
        self._death = death
        self._value = {}
        self._eps = eps

    def process_grad(self, name : str, grad):
        if name not in self._value:
            if self._death:
                self._value[name] = (RunningMeanVar(), RunningMeanVar())
            else:
                self._value[name] = RunningMeanVar()

        new_activation_tensor = isclose(grad, tensor(0.0)).logical_not()
        new_activation_rates = reduce_activation_to_activation_rates(new_activation_tensor, batch=False)

        if self._death:
            activations, death_rates = self._value[name]
            death_rates.update(new_activation_rates.eq(0.0).float().mean().item())
            for act in new_activation_rates:
                activations.update(act.item())
        else:
            activations = self._value[name]
            for act in new_activation_rates:
                activations.update(act.item())



    @property
    def value(self) -> dict[str, Any]:
        return self._value

    def reset(self) -> None:
        self._value = {}
