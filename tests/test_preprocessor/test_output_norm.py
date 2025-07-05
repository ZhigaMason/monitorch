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
def test_sequence_output_norm(module, inp_size, normalize, n_iter, seed):
    onm = OutputNorm(normalize=normalize, inplace=False)
    onr = OutputNorm(normalize=normalize, inplace=True)

    ffg = FeedForwardGatherer(
        module, [onm, onr], 'standalone_test'
    )

    x = torch.zeros(*inp_size)
    norms = []

    torch.manual_seed(seed)
    for _ in range(n_iter):
        x.cauchy_()
        y = module(x)

        norm = vector_norm(y).item()
        if normalize:
            norm /= sqrt(y.numel())

        assert np.isclose(norm, onm.value['standalone_test'][-1])

        norms.append(norm)

    rmv = onr.value['standalone_test']
    assert np.isclose(
        [np.mean(norms), np.var(norms), np.min(norms), np.max(norms)],
        [rmv.mean, rmv.var, rmv.min_, rmv.max_]
    ).all()
