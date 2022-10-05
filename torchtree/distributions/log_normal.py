import math
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
    :param scale_real: standard deviation of the distribution
    """

    def __init__(
        self,
        mean: Union[Tensor, float],
        scale: Union[Tensor, float] = None,
        scale_real: Union[Tensor, float] = None,
        validate_args=None,
    ) -> None:
        if (scale is not None) + (scale_real is not None) != 1:
            raise ValueError("Exactly one of scale or scale_real may be specified.")

        if scale is not None:
            log_mean = log(mean) if isinstance(mean, Number) else mean.log()
            var = scale * scale if isinstance(scale, Number) else scale.pow(2)
            loc = log_mean - var * 0.5
        else:
            temp = 1.0 + scale_real * scale_real / (mean * mean)
            if isinstance(temp, Number):
                loc = math.log(mean / math.sqrt(temp))
                scale = math.sqrt(math.log(temp))
            else:
                loc = torch.log(mean / temp.sqrt())
                scale = temp.log().sqrt()
        super().__init__(loc, scale, validate_args=validate_args)
