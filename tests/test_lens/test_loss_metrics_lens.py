import pytest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from warnings import filterwarnings

from torch.utils.data import TensorDataset, DataLoader
from lenses_testing_utils import generic_lens_test, N_DIM, N_EPOCHS, _generate_xor_data, _xor_ground_truth
from monitorch.inspector import PyTorchInspector
from monitorch.lens import LossMetrics
from monitorch.visualizer import MatplotlibVisualizer

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
    ]
)
def test_implicit_loss_accumulation(module, loss_fn, vizualizer, lens_kwargs):
    inspector = PyTorchInspector(
        lenses = [
            LossMetrics(loss_fn=loss_fn, **lens_kwargs)
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
        ), nn.BCELoss(), 'matplotlib', {'separate_loss_and_metrics' : True}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'tensorboard', {'separate_loss_and_metrics' : True}),
        ( nn.Sequential(
            nn.Linear(N_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ), nn.BCELoss(), 'print', {'separate_loss_and_metrics' : True}),
    ]
)
def test_explicit_loss_accuracy_accumulation(module, loss_fn, vizualizer, lens_kwargs):
    inspector = PyTorchInspector(
        lenses = [
            LossMetrics(loss_fn=loss_fn, **lens_kwargs)
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

    if isinstance(inspector.vizualizer, MatplotlibVisualizer):
        filterwarnings(
            "ignore",
            message=MatplotlibVisualizer.NO_SMALL_TAGS_WARNING,
            category=UserWarning
        )

    raw_data = torch.tensor(_generate_xor_data(N_DIM)).reshape(-1, N_DIM)
    border = 8 * (2 ** N_DIM) // 10
    train_data = raw_data[:border]
    validation_data = raw_data[border:]
    train_loader = DataLoader(
        TensorDataset(train_data), shuffle=True, batch_size=32
    )
    validation_loader = DataLoader(
        TensorDataset(validation_data), shuffle=False, batch_size=32
    )

    for _ in range(N_EPOCHS):
        correct = 0
        for data, in train_loader:
            optimizer.zero_grad()
            y_pred = module(data)
            label = _xor_ground_truth(data)
            loss = loss_fn(y_pred, label)
            inspector.push_loss(loss, train=True, running=lens_kwargs.get('inplace', True))
            inspector.push_metric(
                'train_accuracy',
                (correct == y_pred).float().sum().item(),
                running=lens_kwargs.get('inplace', True)
            )
            optimizer.step()

        with torch.no_grad():
            for data, in validation_loader:
                y_pred = module(data)
                label = _xor_ground_truth(data)
                loss = loss_fn(y_pred, label)
                inspector.push_loss(loss, train=False, running=lens_kwargs.get('inplace', True))
        inspector.tick_epoch()

    if isinstance(inspector.vizualizer, MatplotlibVisualizer):
        fig = inspector.vizualizer.show_fig()
        plt.close(fig) # otherwise figs comsume all the memory due to pytest running tests in parallel
