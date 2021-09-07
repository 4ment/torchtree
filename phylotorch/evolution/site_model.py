from abc import abstractmethod
from typing import Optional, Union

import torch
from torch.distributions import LogNormal

from ..core.abstractparameter import AbstractParameter
from ..core.model import Model
from ..core.parameter import Parameter
from ..core.utils import process_object, register_class
from ..typing import ID


class SiteModel(Model):
    _tag = 'site_model'

    @abstractmethod
    def rates(self) -> torch.Tensor:
        pass

    @abstractmethod
    def probabilities(self) -> torch.Tensor:
        pass


@register_class
class ConstantSiteModel(SiteModel):
    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_)
        self._rate = mu if mu is not None else Parameter(None, torch.ones((1,)))
        self._probability = torch.ones_like(self._rate.tensor)

    def rates(self) -> torch.Tensor:
        return self._rate.tensor

    def probabilities(self) -> torch.Tensor:
        return self._probability

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return self._rate.shape[:-1]

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._rate.cuda(device)
        self._probability = self._probability.cuda(device)

    def cpu(self) -> None:
        self._rate.cpu()
        self._probability = self._probability.cpu()

    @classmethod
    def from_json(cls, data, dic):
        if 'mu' in data:
            mu = process_object(data['mu'], dic)
        else:
            mu = Parameter(None, torch.ones((1,)))
        return cls(data['id'], mu)


@register_class
class InvariantSiteModel(SiteModel):
    def __init__(
        self, id_: ID, invariant: AbstractParameter, mu: AbstractParameter = None
    ) -> None:
        super().__init__(id_)
        self._invariant = invariant
        self._mu = mu
        self._rates = None
        self._probs = None
        self.need_update = True

    @property
    def invariant(self) -> torch.Tensor:
        return self._invariant.tensor

    def update_rates_probs(self, invariant: torch.Tensor):
        self._probs = torch.cat((invariant, 1.0 - invariant), -1)
        self._rates = torch.cat(
            (
                torch.zeros_like(invariant, device=invariant.device),
                1.0 / (1.0 - invariant),
            ),
            -1,
        )
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates_probs(self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates_probs(self.invariant)
            self.need_update = False
        return self._probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return self._invariant.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        invariant = process_object(data['invariant'], dic)
        if 'mu' in data:
            mu = process_object(data['mu'], dic)
        else:
            mu = None
        return cls(id_, invariant, mu)


@register_class
class WeibullSiteModel(SiteModel):
    def __init__(
        self,
        id_: ID,
        shape: AbstractParameter,
        categories: int,
        invariant: AbstractParameter = None,
        mu: AbstractParameter = None,
    ) -> None:
        super().__init__(id_)
        self._shape = shape
        self.categories = categories
        self._invariant = invariant
        self._mu = mu
        self.probs = torch.full(
            (categories,), 1.0 / categories, dtype=self.shape.dtype, device=shape.device
        )
        self._rates = None
        self.need_update = True

    @property
    def shape(self) -> torch.Tensor:
        return self._shape.tensor

    @property
    def invariant(self) -> torch.Tensor:
        return self._invariant.tensor if self._invariant else None

    def update_rates(self, shape: torch.Tensor, invariant: torch.Tensor):
        if invariant:
            cat = self.categories - 1
            quantile = (2.0 * torch.arange(cat, device=shape.device) + 1.0) / (
                2.0 * cat
            )
            self.probs = torch.cat(
                (
                    invariant,
                    torch.full((cat,), (1.0 - invariant) / cat, device=shape.device),
                )
            )
            rates = torch.cat(
                (
                    torch.zeros_like(invariant),
                    torch.pow(-torch.log(1.0 - quantile), 1.0 / shape),
                )
            )
        else:
            quantile = (
                2.0 * torch.arange(self.categories, device=shape.device) + 1.0
            ) / (2.0 * self.categories)
            rates = torch.pow(-torch.log(1.0 - quantile), 1.0 / shape)

        self._rates = rates / (rates * self.probs).sum(-1, keepdim=True)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self.probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return max(
            [parameter.shape[:-1] for parameter in self._parameters.values()],
            key=len,
        )

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda()
        self.need_update = True

    def cpu(self) -> None:
        super().cpu()
        self.need_update = True

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        shape = process_object(data['shape'], dic)
        categories = data['categories']
        invariant = None
        if 'invariant' in data:
            invariant = process_object(data['invariant'], dic)
        if 'mu' in data:
            mu = process_object(data['mu'], dic)
        else:
            mu = None
        return cls(id_, shape, categories, invariant, mu)


class LogNormalSiteModel(SiteModel):
    def __init__(self, id_: ID, scale: AbstractParameter, categories: int = 4) -> None:
        super().__init__(id_)
        self._scale = scale
        self.categories = categories
        self.probs = torch.full((categories,), 1.0 / categories, dtype=self.scale.dtype)
        self.quantile = (2.0 * torch.arange(categories) + 1.0) / (2.0 * categories)
        self._rates = None
        self.update_rates(self.scale)
        self.need_update = True

    @property
    def scale(self) -> torch.Tensor:
        return self._scale.tensor

    def update_rates(self, value):
        rates = LogNormal(-value * value / 2.0, value).icdf(self.quantile)
        self._rates = rates / (rates.sum() * self.probs)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self.scale)
            self.need_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        return self.probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return max(
            [parameter.shape[:-1] for parameter in self._parameters.values()],
            key=len,
        )

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda()
        self.probs.cuda()

    def cpu(self) -> None:
        super().cpu()
        self.probs.cpu()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        scale = process_object(data['scale'], dic)
        categories = int(data['categories'])
        return cls(id_, scale, categories)
