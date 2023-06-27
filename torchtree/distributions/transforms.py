"""Invertible transformations inheriting from torch.distributions.Transform."""
import math
from typing import Union

import torch
from torch.autograd.functional import jacobian
from torch.distributions import Transform, constraints
from torch.nn.functional import softplus

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.utils import register_class


@register_class
class TrilExpDiagonalTransform(Transform):
    r"""Transform a 1D tensor to a triangular tensor. The diagonal of the
    triangular matrix is exponentiated. Useful for variational inference with
    the multivariate normal distribution as the variational distribution and it
    is parameterized with scale_tril, a lower-triangular matrix with positive
    diagonal.

    Example:

        >>> x = torch.tensor([1., 2., 3.])
        >>> y = TrilExpDiagonalTransform()(x)
        >>> y
        tensor([[ 2.7183,  0.0000],
                [ 2.0000, 20.0855]])
        >>> torch.allclose(TrilExpDiagonalTransform().inv(y), x)
        True
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


class CumSumTransform(Transform):
    r"""Transform via the mapping :math:`y_i = \sum_{j=0}^i x_j`.

    >>> x = torch.tensor([1., 2., 3.])
    >>> all(CumSumTransform()(x) == torch.tensor([1., 3., 6.]))
    True
    >>> all(CumSumTransform().inv(torch.tensor([1., 3., 6.])) == x)
    True
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def _call(self, x):
        return x.cumsum(-1)

    def _inverse(self, y):
        return torch.cat((y[..., :1], y[..., 1:] - y[..., :-1]), -1)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)


@register_class
class CumSumExpTransform(Transform):
    r"""Transform via the mapping :math:`y_i = \exp(\sum_{j=0}^i x_j)`."""
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def _call(self, x):
        return x.cumsum(-1).exp()

    def _inverse(self, y):
        y_log = y.log()
        return torch.cat((y_log[..., :1], y_log[..., 1:] - y_log[..., :-1]), -1)

    def log_abs_det_jacobian(self, x, y):
        def f(xx):
            return xx.cumsum(-1).exp()

        # The Jacobian is triangular so we compute the determinant using the diagonal
        if x.dim() == 1:
            jac = jacobian(f, x)
            return torch.diagonal(jac, 0).log().sum()
        else:
            return torch.stack(
                [
                    torch.diagonal(jacobian(f, x[i])).log().sum()
                    for i in range(x.shape[0])
                ]
            )


class SoftPlusTransform(Transform):
    r"""Transform via the mapping :math:`y_i = \log(\exp(x_i) + 1)`."""
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return torch.expm1(y).log()

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)


class CumSumSoftPlusTransform(Transform):
    r"""Transform via the mapping :math:`y_i = \log(\exp(\sum_{j=0}^i x_j) +
    1)`."""
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def _call(self, x):
        return torch.log(x.cumsum(-1).exp() + 1.0)

    def _inverse(self, y):
        y_log = y.log()
        return torch.cat((y_log[..., :1], y_log[..., 1:] - y_log[..., :-1]), -1)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[:-1])


@register_class
class ConvexCombinationTransform(Transform):
    r"""Transform from unconstrained space to constrained space via :math:`y =
    \frac{x}{\sum_{i=1}^K \alpha_i x_i}` in order to satisfy
    :math:`\sum_{i=1}^K \alpha_i y_i = 1` where :math:`\alpha_i \geq 0` and
    :math:`\sum_{i=1}^K \alpha_i = 1`.

    :param weights: weights (sum to 1)
    """
    domain = constraints.positive
    codomain = constraints.positive

    def __init__(self, weights: AbstractParameter, cache_size=0) -> None:
        super(ConvexCombinationTransform, self).__init__(cache_size=cache_size)
        self._weights = weights

    def _call(self, x):
        return x / (x * self._weights.tensor).sum(axis=-1, keepdims=True)

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[:-1])


@register_class
class LogTransform(Transform):
    r"""Transform via the mapping :math:`y = \log(x)`."""
    domain = constraints.positive
    codomain = constraints.real
    bijective = True
    sign = +1

    def _call(self, x):
        return x.log()

    def _inverse(self, y):
        return y.exp()

    def log_abs_det_jacobian(self, x, y):
        return -y


@register_class
class LinearTransform(Transform):
    r"""Transform via the mapping :math:`y = Ax + b`."""
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        A: Union[AbstractParameter, torch.Tensor],
        b: AbstractParameter,
        cache_size=0,
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.A = A
        self.b = b

    def _call(self, x):
        if isinstance(self.A, torch.Tensor):
            if self.A.dim() == 1:
                return self.A * x + self.b.tensor
            else:
                return torch.squeeze(self.A.t() @ x.unsqueeze(-1), -1) + self.b.tensor
        else:
            return (
                torch.squeeze(self.A.tensor.t() @ x.unsqueeze(-1), -1) + self.b.tensor
            )

    def _inverse(self, y):
        if isinstance(self.A, torch.Tensor):
            if self.A.dim() == 1:
                return (y - self.b) / self.A
            else:
                return self.A.t().inverse() @ (y - self.b)
        else:
            return self.A.t().tensor.inverse() @ (y - self.b)

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError
