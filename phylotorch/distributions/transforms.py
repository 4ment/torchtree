import math

import torch
from torch.distributions import Transform


class TrilExpDiagonalTransform(Transform):
    r"""
    Transform a 1D tensor to a triangular tensor. The diagonal of the triangular matrix is exponentiated.
    Useful for variational inference with the multivariate normal distribution as the variational distribution
    and it is parameterized with scale_tril, a lower-triangular matrix with positive diagonal.

    Example:

        >>> x = torch.tensor([1., 2., 3.])
        >>> y = TrilExpDiagonalTransform()(x)
        >>> y
        tensor([[ 2.7183,  0.0000],
                [ 2.0000, 20.0855]])
        >>> TrilExpDiagonalTransform().inv(y)
        tensor([1.0000, 2.0000, 3.0000])
    """
    bijective = True
    sign = +1

    def _call(self, x):
        dim = int((-1 + math.sqrt(1 + 8 * x.shape[0])) / 2)
        tril = torch.zeros((dim, dim), dtype=x.dtype)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        tril[tril_indices[0], tril_indices[1]] = x
        tril[range(dim), range(dim)] = tril.diag().exp()
        return tril

    def _inverse(self, y):
        tril_indices = torch.tril_indices(row=y.shape[0], col=y.shape[0], offset=0)
        x = y.clone()
        x[range(y.shape[0]), range(y.shape[0])] = x.diag().log()
        return x[tril_indices[0], tril_indices[1]]

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError
