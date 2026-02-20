import copy

import numpy as np
import torch

from .running_value import Accumulator, RunningMeanVar


class GeometryComputation:
    """
    An object used for geometry calculation.

    Keeps track of norm (RMS) and optionally adjacent dot product.

    Let :math:`X_1, ..., X_n` be sequence of tensors passed to :meth:`update`.

    Keeps track of :math:`n_k = ||X||_2` or :math:`n_k' = \\frac{1}{\\sqrt{\\dim(X)}}||X||_2` (if `normalize=True`)

    Optionally :math:`r_k = \\frac{X_{k-1} \\cdot X_k}{n_{k-1}n_k + \\epsilon}`

    Parameters
    ----------
    inplace : bool
        Flag indicating use of :class:`RunningMeanVar`
    normalize : bool
        Flag indicating if norm of tensor should be computed as RMS
    dot_product : bool
        Flag indicating computation of adjacent scalar product (correlation), increases memory consumption by storing copy of previous tensor
    eps : float
        Constant used for numerical stability when diving dot product by norms
    """

    def __init__(self, inplace: bool, normalize: bool, dot_product: bool, eps: float):
        self.norm = RunningMeanVar() if inplace else list()
        self.eps = eps
        self.normalize = normalize
        self.dot_product = dot_product
        if dot_product:
            self.prev_norm = 1.0
            self.prev_value = 0
            self.product = RunningMeanVar() if inplace else list()

    def update(self, X: torch.Tensor):
        """
        Performs an update step on norm and optionally product

        Parameters
        ----------
        X : torch.Tensor
            Tensor to use for update
        """
        new_norm = torch.linalg.vector_norm(X)
        self.norm.append((new_norm.item() / np.sqrt(X.numel())) if self.normalize else new_norm.item())

        if self.dot_product:
            new_product = torch.sum(self.prev_value * X) / (self.prev_norm * new_norm + self.eps)
            self.product.append(new_product.item())
            self.prev_norm = new_norm
            self.prev_value = copy.deepcopy(X)

    @property
    def value(self) -> Accumulator | tuple[Accumulator, Accumulator]:
        """
        Accumulated values (either list or :class:`RunningMeanVar`)

        Returns
        -------
        tuple[`norms`, `products`]
            if ``dot_product=True``
        `norms`
            if ``dot_product=False``
        """
        if self.dot_product:
            return self.norm, self.product
        return self.norm
