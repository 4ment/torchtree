"""Normal distribution parametrized by location and precision."""
from math import sqrt
from numbers import Number
from typing import Union

import torch.distributions.normal
from torch import Tensor

from ..core.utils import register_class


@register_class
class Normal(torch.distributions.Normal):
    r"""Normal distribution parametrized by location and precision.

    Creates a normal distribution parameterized by :attr:`loc` and
    :attr:`scale` or :attr:`precision`.

    .. math:: X \sim \mathcal{N}(\mu, 1/\tau)

    where :math:`\tau = 1/ \sigma^2`

    :example:
    >>> x = torch.tensor(0.1)
    >>> scale = torch.tensor(0.1)
    >>> norm1 = Normal(torch.tensor(0.5), precision=1.0/scale**2)
    >>> norm2 = torch.distributions.Normal(torch.tensor(0.5), scale=scale)
    >>> norm1.log_prob(x) == norm2.log_prob(x)
    tensor(True)

    :param float or Tensor loc: mean of the distribution.
    :param float or Tensor scale: standard deviation.
    :param float or Tensor precision: precision.
    """

    def __init__(
        self,
        loc: Union[float, Tensor],
        scale: Union[float, Tensor] = None,
        precision: Union[float, Tensor] = None,
        validate_args=None,
    ) -> None:
        scale_ = None
        if (scale is not None) + (precision is not None) != 1:
            raise ValueError("Exactly one of scale or precision may be specified.")

        if scale is not None:
            scale_ = scale
        elif precision is not None:
            if isinstance(precision, Number):
                scale_ = 1.0 / sqrt(precision)
            else:
                scale_ = 1.0 / torch.sqrt(precision)
        super().__init__(loc, scale_, validate_args=validate_args)
