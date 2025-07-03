import torch
import torch.nn as nn
import pytest
from monitorch.numerical import threshold_for_module

@pytest.mark.parametrize(
    ['module'],
    [
        (nn.CELU(),),
        (nn.ELU(),),
        (nn.GELU(),),
        (nn.Hardshrink(),),
        (nn.Hardsigmoid(),),
        (nn.Hardswish(),),
        (nn.Hardtanh(),),
        (nn.LeakyReLU(),),
        (nn.LogSigmoid(),),
        (nn.Mish(),),
        (nn.PReLU(),),
        (nn.ReLU(),),
        (nn.ReLU6(),),
        (nn.RReLU(),),
        (nn.SELU(),),
        (nn.Sigmoid(),),
        (nn.SiLU(),),
        (nn.Softmax(dim=0),),
        (nn.Softplus(),),
        (nn.Softshrink(),),
        (nn.Softsign(),),
        (nn.Tanh(),),
        (nn.Tanhshrink(),),
        (nn.Threshold(10, 5),),
    ]
)
def test_threshold_for_module(module, n_pts=1000, eps=0.075):
    x = torch.zeros(n_pts, dtype=torch.float).cauchy_().requires_grad_(True)
    y = module(x)
    y.sum().backward()
    assert x.grad is not None
    grad_truth = torch.isclose(x.grad, torch.tensor([0.0]))

    lo, up = threshold_for_module(module)

    print(y)
    passed_threshold = torch.isclose(y, torch.tensor([lo], dtype=torch.float)) | torch.isclose(y, torch.tensor([up], dtype=torch.float))
    print(passed_threshold)
    # TODO rewrite for a statistical test
    assert (grad_truth != passed_threshold).float().mean() < eps
