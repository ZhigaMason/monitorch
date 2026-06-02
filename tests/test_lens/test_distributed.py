import gc
import os
from itertools import combinations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from lenses_testing_utils import train_xor
from torch.nn.parallel import DistributedDataParallel as DDP

from monitorch import lens
from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import AbstractVisualizer

N_DIM = 5
N_EPOCHS = 3


class StateDummyVisualizer(AbstractVisualizer):
    def __init__(self):
        self.was_called = False

    def register_tags(self, main_tag: str, tag_attr) -> None:
        pass

    def plot_numerical_values(
        self,
        epoch: int,
        main_tag: str,
        values_dict,
        ranges_dict,
    ) -> None:
        self.was_called = True

    def plot_probabilities(
        self,
        epoch: int,
        main_tag: str,
        values_dict,
    ) -> None:
        self.was_called = True

    def plot_relations(
        self,
        epoch: int,
        main_tag,
        values_dict,
    ) -> None:
        self.was_called = True


def _worker_test_e2e(rank, world_size, lenses, port):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(
            'gloo',
            init_method=f'tcp://127.0.0.1:{port}',
            rank=rank,
            world_size=world_size,
        )

        # Setup Model
        model = nn.Sequential(
            nn.Linear(N_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        ddp_model = DDP(model)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.BCEWithLogitsLoss()

        # Setup Inspector
        # WARN: string eval: not secure.
        # I tolerate this, since it is a test.
        inspector = PyTorchInspector(
            lenses=lenses,
            module=ddp_model,
            visualizer=StateDummyVisualizer(),
        )

        train_xor(
            inspector=inspector,
            module=ddp_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            n_dim=N_DIM,
            n_epochs=N_EPOCHS,
            push_loss=False,
        )

        # Asserts
        if rank == 0:
            assert inspector.is_main_process
            assert inspector.visualizer.was_called
        else:
            assert not inspector.is_main_process
    finally:
        dist.barrier()
        if 'inspector' in locals():
            del inspector
        if 'ddp_model' in locals():
            del ddp_model
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


bce_loss = nn.BCEWithLogitsLoss()

LENSES_TO_TEST = [
    lens.OutputActivation(),
    lens.OutputGradientGeometry(),
    lens.OutputNorm(),
    lens.ParameterGradientActivation(),
    lens.ParameterGradientGeometry(),
    lens.ParameterNorm(),
]


PARAM_LIST = [(world_size, [lens_]) for world_size in [2, 4] for lens_ in LENSES_TO_TEST] + [(world_size, [lens1, lens2]) for world_size in [2, 4] for lens1, lens2 in combinations(LENSES_TO_TEST, 2)]


@pytest.mark.slow
@pytest.mark.parametrize('param_idx', range(len(PARAM_LIST)))
def test_inspector_e2e(param_idx):
    world_size, lenses = PARAM_LIST[param_idx]

    base_port = 29500
    unique_port = base_port + param_idx

    mp.spawn(
        _worker_test_e2e,
        args=(world_size, lenses, unique_port),
        nprocs=world_size,
        join=True,
    )
