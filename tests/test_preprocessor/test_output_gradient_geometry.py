from copy import deepcopy
from math import sqrt

import pytest
import numpy as np
import torch.nn as nn
import torch
from torch.linalg import vector_norm

from monitorch.preprocessor import OutputGradientGeometryMemory, OutputGradientGeometryRunning
from monitorch.gatherer import BackwardGatherer




def collect_output_grad(l : list):
    def f(module, grad_inp, grad_out):
        l.append(deepcopy(grad_out[0]))
    return f

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
def test_sequence_output_gradient_norm(module, inp_size, normalize, n_iter, seed):
    oggm = OutputGradientGeometryMemory(adj_prod=True, normalize=normalize)
    oggr = OutputGradientGeometryRunning(adj_prod=True, normalize=normalize)

    bg = BackwardGatherer(
        module, [oggm, oggr], 'standalone_test'
    )

    x = torch.zeros(*inp_size)
    norms = [1]
    products = []
    grads = [torch.tensor(0)]

    module.register_full_backward_hook(collect_output_grad(grads))

    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        module(x).square().mean().backward()

        norm = vector_norm(grads[-1]).item()
        if normalize:
            norm /= sqrt(grads[-1].numel())
        prod = (grads[-2] * grads[-1]).sum().item() / (norm * norms[-1])
        if normalize:
            prod /= grads[-1].numel()

        assert np.isclose(norm, oggm.value['standalone_test'][-1][0])

        assert abs(prod) <= (1 + 1e-5)
        assert np.isclose(prod, oggm.value['standalone_test'][-1][1])

        norms.append(norm)
        products.append(prod)

    del norms[0]
    del grads

    rmv = oggr.value['standalone_test'][0]
    assert np.isclose(
        [np.mean(norms), np.var(norms), np.min(norms), np.max(norms)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()
    rmv = oggr.value['standalone_test'][1]
    assert np.isclose(
        [np.mean(products), np.var(products), np.min(products), np.max(products)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()
