from math import sqrt
from copy import deepcopy
import pytest

import torch
import numpy as np
import torch.nn as nn
from torch.linalg import vector_norm

from monitorch.preprocessor import (
        WeightGradientGeometryMemory, WeightGradientGeometryRunning,
        BiasGradientGeometryMemory,   BiasGradientGeometryRunning,
        OutputGradientGeometryMemory, OutputGradientGeometryRunning,
)

from monitorch.gatherer import WeightGradientGatherer, BiasGradientGatherer

def replace_w_grad(tensor):
    def f(module, inp, out):
        module.weight.grad = tensor
    return f

def replace_b_grad(tensor):
    def f(module, inp, out):
        module.bias.grad = tensor
    return f

@pytest.mark.parametrize(
    ['module', 'inp_size', 'grad_w', 'grad_b', 'normalize'],
    [
        (nn.Linear(10, 5),   (10,),       torch.ones(5, 10),      torch.ones(5), False),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), torch.ones(5, 1, 2, 2), torch.ones(5), False),
        (nn.Conv1d(1, 5, 2), (1, 16),     torch.ones(5, 1, 2),    torch.ones(5), False),

        (nn.Linear(10, 5),   (10,),       torch.ones(5, 10),      torch.ones(5), True),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), torch.ones(5, 1, 2, 2), torch.ones(5), True),
        (nn.Conv1d(1, 5, 2), (1, 16),     torch.ones(5, 1, 2),    torch.ones(5), True),

        (nn.Linear(10, 5),   (10,),       torch.rand(5, 10),      torch.rand(5), False),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), torch.rand(5, 1, 2, 2), torch.rand(5), False),
        (nn.Conv1d(1, 5, 2), (1, 16),     torch.rand(5, 1, 2),    torch.rand(5), False),

        (nn.Linear(10, 5),   (10,),       torch.rand(5, 10),      torch.rand(5), True),
        (nn.Conv2d(1, 5, 2), (1, 16, 16), torch.rand(5, 1, 2, 2), torch.rand(5), True),
        (nn.Conv1d(1, 5, 2), (1, 16),     torch.rand(5, 1, 2),    torch.rand(5), True),
    ]
)
def test_artificial_gradient_norm(module, inp_size, grad_w, grad_b, normalize):
    wggm = WeightGradientGeometryMemory(adj_prod=False, normalize=normalize)
    wggr = WeightGradientGeometryRunning(adj_prod=False, normalize=normalize)
    bggm = BiasGradientGeometryMemory(adj_prod=False, normalize=normalize)
    bggr = BiasGradientGeometryRunning(adj_prod=False, normalize=normalize)

    module.register_full_backward_hook(replace_w_grad(grad_w))
    module.register_full_backward_hook(replace_b_grad(grad_b))

    wgg = WeightGradientGatherer(
        module, [wggm, wggr], 'standalone_test'
    )
    bgg = BiasGradientGatherer(
        module, [bggm, bggr], 'standalone_test'
    )

    x = torch.ones(*inp_size)
    y = module(x)
    y.square().mean().backward()

    w_norm = vector_norm(grad_w)
    b_norm = vector_norm(grad_b)
    if normalize:
        w_norm /= sqrt(grad_w.numel())
        b_norm /= sqrt(grad_b.numel())

    assert np.isclose(w_norm, wggm.value['standalone_test'][-1])
    assert np.isclose(w_norm, wggr.value['standalone_test'].mean)

    assert np.isclose(b_norm, bggm.value['standalone_test'][-1])
    assert np.isclose(b_norm, bggr.value['standalone_test'].mean)


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
def test_sequence_gradient_norm(module, inp_size, normalize, n_iter, seed):
    wggm = WeightGradientGeometryMemory(adj_prod=True, normalize=normalize)
    wggr = WeightGradientGeometryRunning(adj_prod=True, normalize=normalize)
    bggm = BiasGradientGeometryMemory(adj_prod=True, normalize=normalize)
    bggr = BiasGradientGeometryRunning(adj_prod=True, normalize=normalize)

    wgg = WeightGradientGatherer(
        module, [wggm, wggr], 'standalone_test'
    )
    bgg = BiasGradientGatherer(
        module, [bggm, bggr], 'standalone_test'
    )

    x = torch.zeros(*inp_size)
    prev_w_grad = 0.0
    prev_b_grad = 0.0
    w_norms = [1]
    w_products = []
    b_norms = [1]
    b_products = []

    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        module(x).square().mean().backward()

        w_norm = vector_norm(module.weight.grad)
        b_norm = vector_norm(module.bias.grad)
        w_prod = (prev_w_grad * module.weight.grad).sum() / (w_norm * w_norms[-1])
        b_prod = (prev_b_grad * module.bias.grad).sum()   / (b_norm * b_norms[-1])
        if normalize:
            w_norm /= sqrt(module.weight.grad.numel())
            b_norm /= sqrt(module.bias.grad.numel())

        assert np.isclose(w_norm, wggm.value['standalone_test'][-1][0])
        assert np.isclose(b_norm, bggm.value['standalone_test'][-1][0])

        assert np.isclose(w_prod, wggm.value['standalone_test'][-1][1])
        assert np.isclose(b_prod, bggm.value['standalone_test'][-1][1])

        w_norms.append(w_norm)
        w_products.append(w_prod)
        b_norms.append(b_norm)
        b_products.append(b_prod)
        prev_w_grad = module.weight.grad
        prev_b_grad = module.bias.grad

    del w_norms[0]
    del b_norms[0]

    rmv = wggr.value['standalone_test'][0]
    assert np.isclose(
        [np.mean(w_norms), np.var(w_norms), np.min(w_norms), np.max(w_norms)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()

    rmv = bggr.value['standalone_test'][0]
    assert np.isclose(
        [np.mean(b_norms), np.var(b_norms), np.min(b_norms), np.max(b_norms)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()

