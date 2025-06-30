import torch
import pytest

from monitorch.numerical import reduce_activation_to_activation_rates

@pytest.mark.parametrize(
    ['tensor', 'act_rates'],
    [
        (
            torch.tensor([[1, 0, 1]]).bool(),
            torch.tensor([1, 0, 1]).float()
        ),
        (
            torch.tensor([[1,1,0,0], [0,0,1,1]]).bool(),
            torch.tensor([0.5, 0.5, 0.5, 0.5])
        ),
        (
           torch.tensor([
                [ [1, 0], [1, 0], [1, 0] ],
                [ [0, 0], [0, 1], [1, 1] ],
                [ [1, 1], [0, 0], [0, 1]]
            ]).bool(),
            torch.tensor([ 0.5, 1/3, 2/3])
        )
    ]
)
def test_batch_activations(tensor : torch.Tensor, act_rates : torch.Tensor):
    res = reduce_activation_to_activation_rates(tensor, batch=True)
    assert torch.allclose(res, act_rates)

@pytest.mark.parametrize(
    ['tensor', 'act_rates'],
    [
        (
            torch.tensor([[1, 0, 1]]).bool(),
            torch.tensor(2/3)
        ),
        (
            torch.tensor([
                [1,1,0,0],
                [0,0,1,1]
            ]).bool(),
            torch.tensor([0.5, 0.5])
        ),
        (
           torch.tensor([
                [ [1, 0], [1, 0], [1, 0] ],
                [ [0, 0], [0, 1], [1, 1] ],
                [ [1, 1], [0, 0], [0, 1]]
            ]).bool(),
            torch.tensor([ 0.5, 0.5, 0.5])
        )
    ]
)
def test_nonbatch_activations(tensor : torch.Tensor, act_rates : torch.Tensor):
    res = reduce_activation_to_activation_rates(tensor, batch=False)
    assert torch.allclose(res, act_rates)
