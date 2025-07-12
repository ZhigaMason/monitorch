import torch
import shutil

from torch.utils.data import TensorDataset, DataLoader
from monitorch.inspector import PyTorchInspector
from warnings import filterwarnings
from monitorch.vizualizer import MatplotlibVizualizer, TensorBoardVizualizer

N_DIM = 10
N_EPOCHS = 3

def generic_lens_test(inspector, module, loss_fn, optimizer):

    if isinstance(inspector.vizualizer, MatplotlibVizualizer):
        filterwarnings(
            "ignore",
            message=MatplotlibVizualizer.NO_SMALL_TAGS_WARNING,
            category=UserWarning
        )
    train_xor(
        inspector,
        module,
        loss_fn,
        optimizer,
        n_dim=N_DIM,
        n_epochs=N_EPOCHS,
        push_loss=False
    )

    if isinstance(inspector.vizualizer, MatplotlibVizualizer):
        inspector.vizualizer.show_fig()
    elif isinstance(inspector.vizualizer, TensorBoardVizualizer):
        # shyte
        filterwarnings("ignore")
        shutil.rmtree("runs", ignore_errors=True)

def train_xor(inspector : PyTorchInspector, module, loss_fn, optimizer, n_dim, n_epochs, push_loss : bool, push_loss_inplace : bool = False):
    raw_data = torch.tensor(_generate_xor_data(n_dim)).reshape(-1, n_dim)
    border = 8 * (2 ** n_dim) // 10
    train_data = raw_data[:border]
    validation_data = raw_data[border:]
    train_loader = DataLoader(
        TensorDataset(train_data), shuffle=True, batch_size=32
    )
    validation_loader = DataLoader(
        TensorDataset(validation_data), shuffle=False, batch_size=32
    )

    for _ in range(n_epochs):
        for data, in train_loader:
            optimizer.zero_grad()
            y_pred = module(data)
            label = _xor_ground_truth(data)
            loss = loss_fn(y_pred, label)
            if push_loss:
                inspector.push_loss(loss, train=True, running=push_loss_inplace)
            optimizer.step()

        with torch.no_grad():
            for data, in validation_loader:
                y_pred = module(data)
                label = _xor_ground_truth(data)
                loss = loss_fn(y_pred, label)
                if push_loss:
                    inspector.push_loss(loss, train=False, running=push_loss_inplace)
        inspector.tick_epoch()


def _generate_xor_data(n_dim) -> list[list[float]]:
    if n_dim < 1:
        return [[]]

    ret = []
    for bin_str in _generate_xor_data(n_dim - 1):
        ret.append(bin_str + [0.0])
        ret.append(bin_str + [1.0])
    return ret

def _xor_ground_truth(data):
    return data.sum(dim=-1).int().logical_and(torch.tensor([1])).float().reshape(-1, 1)
