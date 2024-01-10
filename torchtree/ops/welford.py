from copy import deepcopy

import torch
from torch import Tensor


class WelfordVariance:
    r"""Welford's online method for estimating (co)variance."""

    def __init__(self, mean: Tensor, variance: Tensor, samples=0) -> None:
        self._mean = mean
        self._variance = variance
        self.samples = samples
        if samples > 1:
            self._variance *= samples - 1

    def __copy__(self):
        return type(self)(self._mean, self._variance / (self.samples - 1), self.samples)

    def __deep_copy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self._mean, memo),
                deepcopy(self._variance / (self.samples - 1), memo),
                deepcopy(self.samples, memo),
            )
            memo[id_self] = _copy
        return _copy

    def add_sample(self, x: Tensor) -> None:
        """Add sample to calculate mean and variance.

        .. math:
            m_{k} = m_{k-1} + (x - m_{k-1})/k
            v_{k} = v_{k-1} + (x_k - v_{k-1})(x_k - v_{k})
        """
        self.samples += 1
        diff = x - self._mean
        self._mean += diff / self.samples
        if self._variance.dim() == 1:
            self._variance += (x - self._mean) * diff
        else:
            self._variance += torch.ger(x - self._mean, diff)

    def remove_sample(self, x: Tensor):
        """Remove sample to calculate mean and variance.

        .. math:
            m_{k-1} = (k m_{k} - x)/(k-1)
            v_{k-1} = v_{k} - (x_k - v_{k-1})(x_k - v_{k})
        """
        diff = x - self._mean
        self._mean = ((self._mean * self.samples) - x) / (self.samples - 1)
        self.samples -= 1
        if self._variance.dim() == 1:
            self._variance -= (x - self._mean) * diff
        else:
            self._variance -= torch.ger(x - self._mean, diff)

    def variance(self) -> Tensor:
        return self._variance / (self.samples - 1)

    def mean(self) -> Tensor:
        return self._mean

    def reset(self) -> None:
        self._mean = torch.zeros_like(self._mean)
        self._variance = torch.zeros_like(self._variance)
        self.samples = 0
