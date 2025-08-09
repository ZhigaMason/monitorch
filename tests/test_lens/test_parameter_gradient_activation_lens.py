import pytest
import torch
import torch.nn as nn

from lenses_testing_utils import generic_lens_test, N_DIM
from monitorch.inspector import PyTorchInspector
from monitorch.lens import ParameterGradientActivation

@pytest.mark.smoke
@pytest.mark.parametrize(
    ['module', 'loss_fn', 'visualizer', 'lens_kwargs'],
    [
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}),

        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        ( nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {'inplace' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'inplace' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'inplace' : False}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {'warning_plot' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'warning_plot' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'warning_plot' : False}),

    ]
)
def test_parameter_gradient_activation(module, loss_fn, visualizer, lens_kwargs):

    inspector = PyTorchInspector(
        lenses = [
            ParameterGradientActivation(**lens_kwargs)
        ],
        module = module,
        visualizer = visualizer
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


