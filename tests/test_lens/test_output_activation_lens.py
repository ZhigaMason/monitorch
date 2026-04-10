import pytest
import torch
import torch.nn as nn
from lenses_testing_utils import N_DIM, generic_lens_test

from monitorch.inspector import PyTorchInspector
from monitorch.lens import OutputActivation


def _channel_last_smoke(lens_kwargs, n_epochs=2):
    """Smoke test helper for channel_last format [batch, seq_len, features]."""
    module = nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, N_DIM))
    inspector = PyTorchInspector(lenses=[OutputActivation(**lens_kwargs)], module=module, visualizer='print')
    optimizer = torch.optim.Adam(module.parameters())
    for _ in range(n_epochs):
        for _ in range(3):
            x = torch.randn(8, 5, N_DIM)  # [batch, seq_len, features]
            y = module(x)
            y.abs().mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        inspector.tick_epoch()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ['module', 'loss_fn', 'vizualizer', 'lens_kwargs'],
    [
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'inplace': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'inplace': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'inplace': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'activation': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'activation': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'activation': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'dropout': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'dropout': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'dropout': False}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'include': [nn.Linear]}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'include': [nn.Linear]}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'include': [nn.Linear]}),
    ],
)
def test_output_activation_lens(module, loss_fn, vizualizer, lens_kwargs):

    inspector = PyTorchInspector(lenses=[OutputActivation(**lens_kwargs)], module=module, visualizer=vizualizer)

    optimizer = torch.optim.NAdam(module.parameters())

    generic_lens_test(inspector, module, loss_fn, optimizer)


@pytest.mark.smoke
@pytest.mark.parametrize(
    'lens_kwargs',
    [
        {'channel_last': True},
        {'channel_last': True, 'inplace': False},
        {'channel_last': True, 'warning_plot': False},
        {'channel_last': True, 'include': [nn.Linear]},
    ],
)
def test_output_activation_lens_channel_last(lens_kwargs):
    _channel_last_smoke(lens_kwargs)
