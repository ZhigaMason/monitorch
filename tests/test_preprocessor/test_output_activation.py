from math import pi
import numpy as np
import torch
import torch.nn as nn
import pytest
from monitorch.preprocessor import OutputActivationRunning, OutputActivationMemory
from monitorch.gatherer import FeedForwardGatherer

class DumbModule(nn.Module):

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, _):
        return self.output

@pytest.mark.parametrize(
    ['module', 'X', 'activation'],
    [
        (DumbModule(torch.zeros(10,10)), torch.rand(10, 10), 0.0),
        (DumbModule(torch.ones(10,10)),  torch.rand(10, 10), 1.0),
        (nn.ReLU(),                      torch.tensor([0.6934, 0.4656, 0.6557, 0.7697, 0.2359, 0.2468, 0.5747, 0.5834, 0.6392, 0.8261]) - 0.5, 0.7),
        (nn.ReLU(),                      torch.tensor([-0.978,  1.534,  5.978, -1.841, -1.345, 0.291, -0.549,  3.051, -0.890, -0.496]), 0.4),
        (nn.ReLU(),                      torch.tensor([ -1.216,  -2.504,   0.206,   1.330,   6.213,   0.053,  -0.570,
            1.486,  -8.944,  -0.194,   3.314,  -0.777,  -0.264,   2.458,
            0.432,  -0.051,   7.436,  -0.196,   5.129,  -0.180,  -3.384,
            -0.802,  -7.045,  -5.363,  -1.730,   0.168,   1.602,   2.740,
            1.893, -13.469]), 0.4666666666666667),
        (nn.Sigmoid(),                   torch.tensor([
            -1000, -500, -100, -10, -5, -1, 0, 1, 5, 10, 100, 500, 1000
        ]), 0.5384615384615384)
    ]
)
def test_output_single_activation(module, X, activation):
    oam = OutputActivationMemory(death=False)
    oar = OutputActivationRunning(death=False)
    ffg = FeedForwardGatherer(module, [
            oam, oar
    ], 'standalone_test' )
    _ = module(X)
    assert np.isclose(activation, oam.value['standalone_test'][0])
    assert np.isclose(activation, oar.value['standalone_test'].mean)

@pytest.mark.parametrize(
    ['module', 'activation_tensor_func', 'n_iter', 'seed'],
    [
        (nn.ReLU(),  lambda y: (y > 0),                    100, 0),
        (nn.ReLU6(), lambda y: ((y < 6) & (y > 0)),        100, 0),
        (nn.Tanh(),  lambda y: ((y < pi/2) & (y > -pi/2)), 100, 0),
        (nn.Mish(),  lambda y: (y.abs() > 1e-7),           100, 0),

        (nn.ReLU(),  lambda y: (y > 0),                    100, 42),
        (nn.ReLU6(), lambda y: ((y < 6) & (y > 0)),        100, 42),
        (nn.Tanh(),  lambda y: ((y < pi/2) & (y > -pi/2)), 100, 42),
        (nn.Mish(),  lambda y: (y.abs() > 1e-7),           100, 42),

        (nn.ReLU(),  lambda y: (y > 0),                    500, 0),
        (nn.ReLU6(), lambda y: ((y < 6) & (y > 0)),        500, 0),
        (nn.Tanh(),  lambda y: ((y < pi/2) & (y > -pi/2)), 500, 0),
        (nn.Mish(),  lambda y: (y.abs() > 1e-7),           500, 0),

        (nn.ReLU(),  lambda y: (y > 0),                    500, 42),
        (nn.ReLU6(), lambda y: ((y < 6) & (y > 0)),        500, 42),
        (nn.Tanh(),  lambda y: ((y < pi/2) & (y > -pi/2)), 500, 42),
        (nn.Mish(),  lambda y: (y.abs() > 1e-7),           500, 42),
    ]
)
def test_output_epoch_death_activation(module, activation_tensor_func, n_iter, seed):
    oam = OutputActivationMemory(death=True)
    oar = OutputActivationRunning(death=True)
    ffg = FeedForwardGatherer(module, [
            oam, oar
    ], 'standalone_test' )

    x = torch.rand(100, 100)
    activations = []
    death_tensor = torch.ones_like(x, dtype=torch.bool)

    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        y = module(x)
        act_tensor = activation_tensor_func(y)
        activations.append(act_tensor.float().mean())
        death_tensor &= ~act_tensor
        assert np.isclose(oam.value['standalone_test'][0][-1], activations[-1])
        assert (death_tensor ^ oam.value['standalone_test'][1]).float().mean() < 2/100

    assert (death_tensor ^ oar.value['standalone_test'][1]).float().mean() < 2/100
    rmv = oar.value['standalone_test'][0]
    mean = np.mean(activations)
    var = np.var(activations)
    min_ = np.min(activations)
    max_ = np.max(activations)
    assert np.isclose([mean, var, min_, max_], [rmv.mean, rmv.var, rmv.min_, rmv.max_]).all()
