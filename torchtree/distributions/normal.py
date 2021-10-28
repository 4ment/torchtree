from math import sqrt
from numbers import Number
from typing import Union

import torch.distributions.normal
from torch import Tensor

from ..core.utils import register_class


@register_class
class Normal(torch.distributions.Normal):
    """
    Creates a normal distribution parameterized by :attr:`loc` and
     :attr:`scale` or :attr:`precision`.

    :param loc: mean of the distribution (often referred to as mu)
    :param scale: standard deviation of the distribution (often referred to as sigma)
    :param precision: precision of the distribution (precision = 1/scale^2)
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
