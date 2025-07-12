import pytest
import torch
import torch.nn as nn

from lenses_testing_utils import generic_lens_test, N_DIM
from monitorch.inspector import PyTorchInspector
from monitorch.lens import ParameterNorm

@pytest.mark.smoke
@pytest.mark.parametrize(
    ['module', 'loss_fn', 'vizualizer', 'lens_kwargs'],
    [
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}),

        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {'comparison_plot' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'comparison_plot' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'comparison_plot' : False}),
    ]
)
def test_parameter_norm_lens(module, loss_fn, vizualizer, lens_kwargs):

    inspector = PyTorchInspector(
        lenses = [
            ParameterNorm(**lens_kwargs)
        ],
        module = module,
        vizualizer = vizualizer
    )

    optimizer = torch.optim.NAdam(
        module.parameters()
    )

    generic_lens_test(
        inspector,
        module,
        loss_fn,
        optimizer
    )


