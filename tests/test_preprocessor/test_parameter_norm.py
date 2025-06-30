
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
import pytest


from monitorch.preprocessor import ParameterNormMemory, ParameterNormRunning
from monitorch.gatherer import EpochModuleGatherer

@pytest.mark.parametrize(
    ['module', 'inp_size', 'normalize', 'n_iter', 'seed'],
    [
        (nn.Linear(10, 5),   (10,),       False, 100, 0),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), False, 100, 0),
        (nn.Conv1d(1, 5, 2), (1, 16),     False, 100, 0),

        (nn.Linear(10, 5),   (10,),       True,  100, 0),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), True,  100, 0),
        (nn.Conv1d(1, 5, 2), (1, 16),     True,  100, 0),

        (nn.Linear(10, 5),   (10,),       False, 100, 42),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), False, 100, 42),
        (nn.Conv1d(1, 5, 2), (1, 16),     False, 100, 42),

        (nn.Linear(10, 5),   (10,),       True,  100, 42),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), True,  100, 42),
        (nn.Conv1d(1, 5, 2), (1, 16),     True,  100, 42),
    ]
)
def test_norm_parameter(module, inp_size, normalize, n_iter, seed):
    pnm = ParameterNormMemory(['weight', 'bias'], normalize=normalize)
    pnr = ParameterNormRunning(['weight', 'bias'], normalize=normalize)

    emg = EpochModuleGatherer(
        module, [pnm, pnr], 'standalone_test'
    )

    x = torch.zeros(*inp_size)
    w_norms = []
    b_norms = []

    sgd = torch.optim.SGD(module.parameters())
    torch.manual_seed(seed)

    for _ in range(n_iter):
        x.cauchy_()
        sgd.zero_grad()
        module(x).square().mean().backward()

        w_norm = torch.linalg.vector_norm(module.weight).item()
        b_norm = torch.linalg.vector_norm(module.bias).item()
        if normalize:
            w_norm /= sqrt(module.weight.numel())
            b_norm /= sqrt(module.bias.numel())

        w_norms.append(w_norm)
        b_norms.append(b_norm)
        emg()
        sgd.step()

    assert np.allclose(
        w_norms, pnm.value['standalone_test']['weight']
    )
    assert np.allclose(
        b_norms, pnm.value['standalone_test']['bias']
    )

    rmv = pnr.value['standalone_test']['weight']
    assert np.allclose(
            [np.mean(w_norms), np.var(w_norms), np.min(w_norms), np.max(w_norms)],
            [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )
    rmv = pnr.value['standalone_test']['bias']
    assert np.allclose(
            [np.mean(b_norms), np.var(b_norms), np.min(b_norms), np.max(b_norms)],
            [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    )
