import pytest
import torch
import torch.nn as nn

from lenses_testing_utils import generic_lens_test, N_DIM
from monitorch.inspector import PyTorchInspector
from monitorch.lens import OutputActivation

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
        ), nn.BCELoss(), 'matplotlib', {'activation' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'activation' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'activation' : False}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {'dropout' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'dropout' : False}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'dropout' : False}),

        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'matplotlib', {'include' : [nn.Linear]}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'include' : [nn.Linear]}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'include' : [nn.Linear]}),
    ]
)
def test_output_activation_lens(module, loss_fn, vizualizer, lens_kwargs):

    inspector = PyTorchInspector(
        lenses = [
            OutputActivation(**lens_kwargs)
        ],
        module = module,
        visualizer = vizualizer
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


