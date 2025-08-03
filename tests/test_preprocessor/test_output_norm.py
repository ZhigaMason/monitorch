from math import sqrt

import pytest
import numpy as np
import torch.nn as nn
import torch
from torch.linalg import vector_norm

from monitorch.preprocessor import OutputNorm
from monitorch.gatherer import FeedForwardGatherer


@pytest.mark.parametrize(
    ['module', 'inp_size', 'normalize', 'n_iter', 'seed'],
    [
        (nn.Linear(10, 5),   (32, 10,),       False, 100, 0),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False, 100, 0),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     False, 100, 0),

        (nn.Linear(10, 5),   (32, 10,),       True,  100, 0),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True,  100, 0),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     True,  100, 0),

        (nn.Linear(10, 5),   (32, 10,),       False, 100, 42),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False, 100, 42),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     False, 100, 42),

        (nn.Linear(10, 5),   (32, 10,),       True,  100, 42),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True,  100, 42),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     True,  100, 42),
    ]
)
def test_sequence_output_norm(module, inp_size, normalize, n_iter, seed):
    onm = OutputNorm(normalize=normalize, inplace=False, record_no_grad=False)
    onr = OutputNorm(normalize=normalize, inplace=True, record_no_grad=False)

    ffg = FeedForwardGatherer(
        module, [onm, onr], 'standalone_test'
    )

    x = torch.zeros(*inp_size)
    norms = []

    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        y = module(x)

        norm = vector_norm(y.flatten(1, -1), dim=-1).mean().item()
        if normalize:
            norm /= sqrt(y[0].numel())

        assert np.isclose(norm, onm.value['standalone_test'][-1])

        norms.append(norm)

    rmv = onr.value['standalone_test']
    assert np.isclose(
        [np.mean(norms), np.var(norms), np.min(norms), np.max(norms)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()

@pytest.mark.parametrize(
    ['module', 'inp_size', 'record_no_grad'],
    [
        (nn.Linear(10, 5),   (32, 10,),       False,),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False,),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     False,),

        (nn.Linear(10, 5),   (32, 10,),       True, ),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True, ),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     True, ),

        (nn.Linear(10, 5),   (32, 10,),       False,),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False,),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     False,),

        (nn.Linear(10, 5),   (32, 10,),       True, ),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True, ),
        (nn.Conv1d(1, 5, 2), (32, 1, 16),     True, ),
    ]
)
def test_record_no_grad(module, inp_size, record_no_grad):
    onm = OutputNorm(normalize=False, inplace=False, record_no_grad=record_no_grad)
    onr = OutputNorm(normalize=False, inplace=True, record_no_grad=record_no_grad)

    ffg = FeedForwardGatherer(
        module, [onm, onr], 'standalone_test'
    )

    with torch.no_grad():
        x = torch.rand(*inp_size)
        module(x)

    if record_no_grad:
        assert 'standalone_test' in onm.value
        assert 'standalone_test' in onr.value
    else:
        assert 'standalone_test' not in onm.value
        assert 'standalone_test' not in onr.value
