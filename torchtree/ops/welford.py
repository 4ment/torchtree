import torch
from torch import Tensor


class WelfordVariance:
    r"""Welford's online method for estimating (co)variance."""

    def __init__(self, mean: Tensor, variance: Tensor, samples=0) -> None:
        self._mean = mean
        self._variance = variance
        self.samples = samples
        if samples > 1:
            self._variance * (samples - 1)

    def add_sample(self, x: Tensor) -> None:
        self.samples += 1
        diff = x - self._mean
        self._mean += diff / self.samples
        if self._variance.dim() == 1:
            self._variance += (x - self._mean) * diff
        else:
            self._variance += torch.ger(x - self._mean, diff)

    def variance(self) -> Tensor:
        return self._variance / (self.samples - 1)

    def mean(self) -> Tensor:
        return self._mean

    def reset(self) -> None:
        self._mean = torch.zeros_like(self._mean)
        self._variance = torch.zeros_like(self._variance)
        self.samples = 0
