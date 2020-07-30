from abc import abstractmethod

import torch
from torch.distributions import LogNormal

from phylotorch.model import Model


class SiteModel(Model):
    @abstractmethod
    def update(self, value):
        pass

    @abstractmethod
    def rates(self):
        pass

    @abstractmethod
    def probabilities(self):
        pass


class ConstantSiteModel(SiteModel):

    def __init__(self):
        SiteModel.__init__(self)

    def update(self, value):
        pass

    def rates(self):
        return torch.tensor([1.0], dtype=torch.float64)

    def probabilities(self):
        return torch.tensor([1.0], dtype=torch.float64)


class WeibullSiteModel(SiteModel):

    def __init__(self, shape, categories=4):
        SiteModel.__init__(self)
        self.shape_key, self.shape = shape
        self.categories = categories
        self.probs = torch.full((categories,), 1.0 / categories, dtype=self.shape.dtype)
        self.quantile = (2.0 * torch.arange(categories) + 1.0) / (2.0 * categories)
        self._rates = None
        self.update_rates(self.shape)

    def update_rates(self, value):
        rates = torch.pow(-torch.log(1.0 - self.quantile), 1.0 / value)
        self._rates = rates / (rates.sum() * self.probs)

    def update(self, value):
        if isinstance(value, dict):
            if self.shape_key in value:
                self.update_rates(value[self.shape_key])
        else:
            self.update_rates(value)

    def rates(self):
        return self._rates

    def probabilities(self):
        return self.probs


class LogNormalSiteModel(SiteModel):

    def __init__(self, scale, categories=4):
        SiteModel.__init__(self)
        self.scale_key, self.scale = scale
        self.categories = categories
        self.probs = torch.full((categories,), 1.0 / categories, dtype=self.scale.dtype)
        self.quantile = (2.0 * torch.arange(categories) + 1.0) / (2.0 * categories)
        self._rates = None
        self.update_rates(self.scale)

    def update_rates(self, value):
        rates = LogNormal(-value * value / 2., value).icdf(self.quantile)
        self._rates = rates / (rates.sum() * self.probs), self.probs

    def update(self, value):
        if isinstance(value, dict):
            if self.scale_key in value:
                self.update_rates(value[self.scale_key])
        else:
            self.update_rates(value)

    def rates(self):
        return self._rates

    def probabilities(self):
        return self.probs
