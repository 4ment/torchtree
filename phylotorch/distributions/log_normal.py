from math import log
from numbers import Number
from typing import Union

import torch.distributions
from torch import Tensor

from ..core.utils import register_class


@register_class
class LogNormal(torch.distributions.LogNormal):
    """
    Creates a lognormal distribution parameterized by :attr:`mean` and
    :attr:`scale`.

    :param mean: mean of the distribution
    :param scale: standard deviation of log of the distribution
    """

    def __init__(
        self,
        mean: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args=None,
    ) -> None:
        log_mean = log(mean) if isinstance(mean, Number) else mean.log()
        var = scale * scale if isinstance(scale, Number) else scale.pow(2)
        loc = log_mean - var * 0.5
        super().__init__(loc, scale, validate_args=validate_args)
