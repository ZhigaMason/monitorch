from warnings import filterwarnings

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from lenses_testing_utils import N_DIM, N_EPOCHS, _generate_xor_data, _xor_ground_truth, generic_lens_test
from torch.utils.data import DataLoader, TensorDataset

from monitorch.inspector import PyTorchInspector
from monitorch.lens import LossMetrics
from monitorch.visualizer import MatplotlibVisualizer


@pytest.mark.smoke
@pytest.mark.parametrize(
    ['module', 'loss_fn', 'visualizer', 'lens_kwargs'],
    [
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}),
    ],
)
def test_implicit_loss_accumulation(module, loss_fn, visualizer, lens_kwargs):
    inspector = PyTorchInspector(lenses=[LossMetrics(loss_fn=loss_fn, **lens_kwargs)], module=module, visualizer=visualizer)

    optimizer = torch.optim.NAdam(module.parameters())

    generic_lens_test(inspector, module, loss_fn, optimizer)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ['module', 'loss_fn', 'visualizer', 'lens_kwargs', 'do_accuracy'],
    [
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'matplotlib', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'tensorboard', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'print', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'separate_loss_and_metrics': True}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'separate_loss_and_metrics': True}, True),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'separate_loss_and_metrics': True}, True),
        # no accuracy login
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'matplotlib', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'tensorboard', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1)), nn.L1Loss(), 'print', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'matplotlib', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'tensorboard', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1, bias=False)), nn.L1Loss(), 'print', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'matplotlib', {'separate_loss_and_metrics': True}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'tensorboard', {'separate_loss_and_metrics': True}, False),
        (nn.Sequential(nn.Linear(N_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()), nn.BCELoss(), 'print', {'separate_loss_and_metrics': True}, False),
    ],
)
def test_explicit_loss_accuracy_accumulation(module, loss_fn, visualizer, lens_kwargs, do_accuracy):
    inspector = PyTorchInspector(
        lenses=[
            LossMetrics(
                loss_fn=loss_fn,
                metrics=['train_accuracy'],
                **lens_kwargs,
            )
        ],
        module=module,
        visualizer=visualizer,
    )

    optimizer = torch.optim.NAdam(module.parameters())

    generic_lens_test(inspector, module, loss_fn, optimizer)

    if isinstance(inspector.visualizer, MatplotlibVisualizer):
        filterwarnings('ignore', message=MatplotlibVisualizer._NO_SMALL_TAGS_WARNING, category=UserWarning)
        filterwarnings('ignore', message='Empty plot', category=UserWarning)

    raw_data = torch.tensor(_generate_xor_data(N_DIM)).reshape(-1, N_DIM)
    border = 8 * (2**N_DIM) // 10
    train_data = raw_data[:border]
    validation_data = raw_data[border:]
    train_loader = DataLoader(TensorDataset(train_data), shuffle=True, batch_size=32)
    validation_loader = DataLoader(TensorDataset(validation_data), shuffle=False, batch_size=32)

    for _ in range(N_EPOCHS):
        correct = 0
        for (data,) in train_loader:
            optimizer.zero_grad()
            y_pred = module(data)
            label = _xor_ground_truth(data)
            loss = loss_fn(y_pred, label)
            inspector.push_loss(loss, train=True, running=lens_kwargs.get('inplace', True))
            if do_accuracy:
                inspector.push_metric('train_accuracy', (correct == y_pred).float().sum().item(), running=lens_kwargs.get('inplace', True))
            optimizer.step()

        with torch.no_grad():
            for (data,) in validation_loader:
                y_pred = module(data)
                label = _xor_ground_truth(data)
                loss = loss_fn(y_pred, label)
                inspector.push_loss(loss, train=False, running=lens_kwargs.get('inplace', True))
        inspector.tick_epoch()

    if isinstance(inspector.visualizer, MatplotlibVisualizer):
        fig = inspector.visualizer.show_fig()
        plt.close(fig)  # otherwise figs comsume all the memory due to pytest running tests in parallel
