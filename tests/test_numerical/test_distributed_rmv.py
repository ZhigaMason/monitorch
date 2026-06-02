import os

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from monitorch.numerical import RunningMeanVar


def _worker_test_rmv_sync(rank, world_size, global_data):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    rmv = RunningMeanVar()

    # Give Rank 0 and Rank 1 different data
    local_data = global_data[rank::world_size]
    for val in local_data:
        rmv.update(val)

    rmv.start_sync(dst_rank=0)
    rmv.finish_sync()

    # Rank 0 should now hold the global stats for [1, 2, 3, 4, 5, 6]
    if rank == 0:
        print(rmv)
        assert np.allclose(rmv.count, len(global_data))
        assert np.allclose(rmv.mean, np.mean(global_data))
        assert np.allclose(rmv.var, np.var(global_data, ddof=0))
        assert np.allclose(rmv.min_, np.min(global_data))
        assert np.allclose(rmv.max_, np.max(global_data))

    dist.destroy_process_group()


@pytest.mark.parametrize(
    ['world_size', 'global_data'],
    [
        (2, [1, 2, 3, 4, 5, 6]),
        (3, [1, 2, 3, 4, 5, 6]),
        (4, [1, 2, 3, 4, 5, 6]),
        (8, list(np.random.standard_t(df=1, size=(250,)))),
    ],
)
def test_distributed_rmv(world_size, global_data):
    # Spawn 2 processes to run the test worker
    mp.spawn(_worker_test_rmv_sync, args=(world_size, global_data), nprocs=world_size, join=True)
