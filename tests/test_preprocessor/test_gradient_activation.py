import torch
import numpy as np
import torch.nn as nn
import pytest

from monitorch.preprocessor import GradientActivation, GradientActivation
from monitorch.gatherer import WeightGradientGatherer, BiasGradientGatherer
from monitorch.numerical import reduce_activation_to_activation_rates


@pytest.mark.parametrize(
    ['module', 'inp'],
    [
        (
            nn.Linear(10, 5),
            torch.ones(20, 10)
        ),
        (
            nn.Conv1d(1, 3, 2),
            torch.ones(20, 1, 10)
        ),
        (
            nn.Conv2d(1, 3, 2),
            torch.ones(20, 1, 10, 10)
        ),
    ]
)
def test_one_pass_gradient_activation(module, inp):
    wgam = GradientActivation(death=True, inplace=False)
    wgar = GradientActivation(death=True, inplace=True)
    bgam = GradientActivation(death=True, inplace=False)
    bgar = GradientActivation(death=True, inplace=True)

    wgg = WeightGradientGatherer(
        module, [wgam, wgar], 'standalone_test'
    )

    bgg = BiasGradientGatherer(
        module, [bgam, bgar], 'standalone_test'
    )

    module(inp).square().mean().backward()

    w_act = reduce_activation_to_activation_rates(torch.isclose(module.weight.grad, torch.tensor(0.0)).logical_not(), batch=False).numpy()
    w_death = (w_act == 0).mean()
    b_act = reduce_activation_to_activation_rates(torch.isclose(module.bias.grad,   torch.tensor(0.0)).logical_not(), batch=False).numpy()
    b_death = (b_act == 0).mean()

    assert np.allclose(w_act, wgam.value['standalone_test'][0])
    assert np.isclose(w_death, wgam.value['standalone_test'][1])
    assert np.allclose(b_act, bgam.value['standalone_test'][0])
    assert np.isclose(b_death, bgam.value['standalone_test'][1])

    rmv = wgar.value['standalone_test'][0]
    assert [
        [np.mean(w_act), np.var(w_act), np.min(w_act), np.max(w_act)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ]
    rmv = wgar.value['standalone_test'][1]
    assert np.allclose(
        [w_death, w_death, w_death],
        [rmv.mean, rmv.min_, rmv.max_]
    )

    rmv = bgar.value['standalone_test'][0]
    assert [
        [np.mean(b_act), np.var(b_act), np.min(b_act), np.max(b_act)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ]
    rmv = bgar.value['standalone_test'][1]
    assert np.allclose(
        [b_death, b_death, b_death],
        [rmv.mean, rmv.min_, rmv.max_]
    )


@pytest.mark.parametrize(
    ['module', 'inp_size', 'n_iter', 'seed'],
    [
        (nn.Linear(10, 5),   (10,),       50, 0),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), 50, 0),
        (nn.Conv1d(1, 5, 2), (1, 16),     50, 0),

        (nn.Linear(10, 5),   (10,),       50, 0),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), 50, 0),
        (nn.Conv1d(1, 5, 2), (1, 16),     50, 0),

        (nn.Linear(10, 5),   (10,),       50, 42),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), 50, 42),
        (nn.Conv1d(1, 5, 2), (1, 16),     50, 42),

        (nn.Linear(10, 5),   (10,),       50, 42),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), 50, 42),
        (nn.Conv1d(1, 5, 2), (1, 16),     50, 42),
    ]
)
def test_sequence_gradient_activation(module, inp_size, n_iter, seed):
    wgam = GradientActivation(death=True, inplace=False)
    wgar = GradientActivation(death=True, inplace=True)
    bgam = GradientActivation(death=True, inplace=False)
    bgar = GradientActivation(death=True, inplace=True)

    wgg = WeightGradientGatherer(
        module, [wgam, wgar], 'standalone_test'
    )

    bgg = BiasGradientGatherer(
        module, [bgam, bgar], 'standalone_test'
    )

    x = torch.zeros(*inp_size)

    w_acts = []
    w_deathes = []
    b_acts = []
    b_deathes = []

    sgd = torch.optim.SGD(module.parameters())
    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        sgd.zero_grad()
        module(x).square().mean().backward()

        w_act = reduce_activation_to_activation_rates(torch.isclose(module.weight.grad, torch.tensor(0.0)).logical_not(), batch=False).numpy()
        w_death = (w_act == 0).mean()
        b_act = reduce_activation_to_activation_rates(torch.isclose(module.bias.grad,   torch.tensor(0.0)).logical_not(), batch=False).numpy()
        b_death = (b_act == 0).mean()

        w_acts.extend(w_act.tolist())
        w_deathes.append(w_death)
        b_acts.extend(b_act.tolist())
        b_deathes.append(b_death)

        sgd.step()

    assert np.allclose(
        w_acts, wgam.value['standalone_test'][0]
    )
    assert np.allclose(
        w_deathes, wgam.value['standalone_test'][1]
    )
    assert np.allclose(
        b_acts, bgam.value['standalone_test'][0]
    )
    assert np.allclose(
        b_deathes, bgam.value['standalone_test'][1]
    )

    rmv = wgar.value['standalone_test'][0]
    assert np.allclose(
        [np.mean(w_acts), np.var(w_acts), np.min(w_acts), np.max(w_acts)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )
    rmv = wgar.value['standalone_test'][1]
    assert np.allclose(
        [np.mean(w_deathes), np.var(w_deathes), np.min(w_deathes), np.max(w_deathes)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )

    rmv = bgar.value['standalone_test'][0]
    assert np.allclose(
        [np.mean(b_acts), np.var(b_acts), np.min(b_acts), np.max(b_acts)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )
    rmv = bgar.value['standalone_test'][1]
    assert np.allclose(
        [np.mean(b_deathes), np.var(b_deathes), np.min(b_deathes), np.max(b_deathes)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )
