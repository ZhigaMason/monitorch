from math import sqrt

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.linalg import vector_norm

from monitorch.gatherer import FeedForwardGatherer
from monitorch.inspector.inspector_state import InspectorState
from monitorch.preprocessor import OutputNorm


@pytest.mark.parametrize(
    ['module', 'inp_size', 'normalize', 'n_iter', 'seed'],
    [
        (nn.Linear(10, 5), (32, 10), False, 100, 0),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False, 100, 0),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), False, 100, 0),
        (nn.Linear(10, 5), (32, 10), True, 100, 0),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True, 100, 0),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), True, 100, 0),
        (nn.Linear(10, 5), (32, 10), False, 100, 42),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False, 100, 42),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), False, 100, 42),
        (nn.Linear(10, 5), (32, 10), True, 100, 42),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True, 100, 42),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), True, 100, 42),
    ],
)
def test_sequence_output_norm(module, inp_size, normalize, n_iter, seed):
    onm = OutputNorm(normalize=normalize, inplace=False, record_eval=False, evaluation_from_grad=True)
    onr = OutputNorm(normalize=normalize, inplace=True, record_eval=False, evaluation_from_grad=True)

    ffg = FeedForwardGatherer(module, [onm, onr], 'standalone_test', InspectorState())  # noqa: F841

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
    assert np.isclose([np.mean(norms), np.var(norms), np.min(norms), np.max(norms)], [rmv.mean, rmv.var, rmv.min_, rmv.max_]).all()


@pytest.mark.parametrize(
    ['inp_size', 'normalize', 'n_iter', 'seed'],
    [
        ((32, 10, 5), False, 100, 0),
        ((32, 10, 5), True, 100, 0),
        ((32, 10, 5), False, 100, 42),
        ((32, 7, 4, 5), False, 100, 0),
        ((32, 7, 4, 5), True, 100, 42),
    ],
)
def test_channel_last_output_norm(inp_size, normalize, n_iter, seed):
    # channel_last format is [batch, seq_len, ..., features]; the norm computation
    # (flatten all non-batch dims, take L2 norm) is equivalent to channel_first.
    module = nn.Identity()
    onm = OutputNorm(normalize=normalize, inplace=False, record_eval=False, channel_last=True, evaluation_from_grad=True)
    onr = OutputNorm(normalize=normalize, inplace=True, record_eval=False, channel_last=True, evaluation_from_grad=True)

    ffg = FeedForwardGatherer(module, [onm, onr], 'standalone_test', InspectorState())  # noqa: F841

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
    assert np.isclose([np.mean(norms), np.var(norms), np.min(norms), np.max(norms)], [rmv.mean, rmv.var, rmv.min_, rmv.max_]).all()


@pytest.mark.parametrize(
    ['module', 'inp_size', 'record_eval'],
    [
        (nn.Linear(10, 5), (32, 10), False),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), False),
        (nn.Linear(10, 5), (32, 10), True),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), True),
        (nn.Linear(10, 5), (32, 10), False),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), False),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), False),
        (nn.Linear(10, 5), (32, 10), True),
        (nn.Conv2d(1, 5, 2), (32, 1, 16, 16), True),
        (nn.Conv1d(1, 5, 2), (32, 1, 16), True),
    ],
)
def test_record_eval(module, inp_size, record_eval):
    onm = OutputNorm(normalize=False, inplace=False, record_eval=record_eval, evaluation_from_grad=True)
    onr = OutputNorm(normalize=False, inplace=True, record_eval=record_eval, evaluation_from_grad=True)

    ffg = FeedForwardGatherer(module, [onm, onr], 'standalone_test', InspectorState())  # noqa: F841

    with torch.no_grad():
        x = torch.rand(*inp_size)
        module(x)

    if record_eval:
        assert 'standalone_test' in onm.value
        assert 'standalone_test' in onr.value
    else:
        assert 'standalone_test' not in onm.value
        assert 'standalone_test' not in onr.value
