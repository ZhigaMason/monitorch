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

from monitorch.gatherer import BackwardGatherer

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
def test_artificial_gradient_norm(module, inp_size, grad_w, grad_b, normalize, eps=1e-7):
    wggm = WeightGradientGeometryMemory(adj_prod=False, normalize=normalize)
    wggr = WeightGradientGeometryRunning(adj_prod=False, normalize=normalize)
    bggm = BiasGradientGeometryMemory(adj_prod=False, normalize=normalize)
    bggr = BiasGradientGeometryRunning(adj_prod=False, normalize=normalize)
    bg = BackwardGatherer([
        wggm, wggr, bggm, bggr
    ], 'standalone_test')
    module.register_full_backward_hook(replace_w_grad(deepcopy(grad_w)))
    module.register_full_backward_hook(replace_b_grad(deepcopy(grad_b)))
    module.register_full_backward_hook(bg)

    x = torch.ones(*inp_size)
    y = module(x)
    y.square().mean().backward()

    w_norm = vector_norm(grad_w)
    b_norm = vector_norm(grad_b)
    if normalize:
        w_norm /= sqrt(grad_w.numel())
        b_norm /= sqrt(grad_b.numel())

    assert abs(w_norm - wggm.value['standalone_test'][-1]) < eps
    assert abs(w_norm - wggr.value['standalone_test'].mean) < eps

    assert abs(b_norm - bggm.value['standalone_test'][-1]) < eps
    assert abs(b_norm - bggr.value['standalone_test'].mean) < eps


@pytest.mark.parametrize(
    ['module', 'inp_size', 'normalize', 'n_iter'],
    [
        (nn.Linear(10, 5), (10,), False, 100)
    ]
)
def test_sequence_gradient_norm(module, inp_size, normalize, n_iter, eps=1e-7):
    wggm = WeightGradientGeometryMemory(adj_prod=True, normalize=normalize)
    wggr = WeightGradientGeometryRunning(adj_prod=True, normalize=normalize)
    bggm = BiasGradientGeometryMemory(adj_prod=True, normalize=normalize)
    bggr = BiasGradientGeometryRunning(adj_prod=True, normalize=normalize)
    bg = BackwardGatherer([
        wggm, wggr, bggm, bggr
    ], 'standalone_test')
    module.register_full_backward_hook(bg)

    x = torch.zeros(*inp_size)
    prev_w_grad = 0.0
    prev_b_grad = 0.0
    w_norms = [1]
    w_products = []
    b_norms = [1]
    b_products = []

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

        assert abs(w_norm - wggm.value['standalone_test'][0][-1]) < eps
        assert abs(b_norm - bggm.value['standalone_test'][0][-1]) < eps

        assert abs(w_prod - wggm.value['standalone_test'][1][-1]) < eps
        assert abs(b_prod - bggm.value['standalone_test'][1][-1]) < eps

        w_norms.append(w_norm)
        w_products.append(w_prod)
        b_norms.append(b_norm)
        b_products.append(b_prod)

    del w_norms[0]
    del b_norms[0]

    assert abs(np.mean(w_norms) - wggr.value['standalone_test'][0].mean) < eps

