import torch
import copy
import numpy as np
from .running_value import RunningMeanVar, Accumulator

class GeometryComputation:

    def __init__(self, inplace : bool, normalize : bool, dot_product : bool, eps : float):
        self.norm = RunningMeanVar() if inplace else list()
        self.eps = eps
        self.normalize = normalize
        self.dot_product = dot_product
        if dot_product:
            self.prev_norm = 1.0
            self.prev_value = 0
            self.product = RunningMeanVar() if inplace else list()

    def update(self, X : torch.Tensor):
        new_norm = torch.linalg.vector_norm(X)
        self.norm.append((new_norm.item() / np.sqrt(X.numel())) if self.normalize else new_norm.item())

        if self.dot_product:
            new_product = torch.sum(self.prev_value * X) / (self.prev_norm * new_norm + self.eps)
            self.product.append(new_product.item())
            self.prev_norm = new_norm
            self.prev_value = copy.deepcopy(X)

    @property
    def value(self) -> Accumulator|tuple[Accumulator, Accumulator]:
        if self.dot_product:
            return self.norm, self.product
        return self.norm
