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

    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_)
        self._mu = mu

    @abstractmethod
    def rates(self) -> torch.Tensor:
        pass

    @abstractmethod
    def probabilities(self) -> torch.Tensor:
        pass


@register_class
class ConstantSiteModel(SiteModel):
    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_, mu)
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
            mu = None
        return cls(data['id'], mu)


@register_class
class InvariantSiteModel(SiteModel):
    def __init__(
        self, id_: ID, invariant: AbstractParameter, mu: AbstractParameter = None
    ) -> None:
        super().__init__(id_, mu)
        self._invariant = invariant
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


class UnivariateDiscretizedSiteModel(SiteModel):
    def __init__(
        self,
        id_: ID,
        parameter: AbstractParameter,
        categories: int,
        invariant: AbstractParameter = None,
        mu: AbstractParameter = None,
    ) -> None:
        super().__init__(id_, mu)
        self._parameter = parameter
        self._categories = categories
        self._invariant = invariant
        self.probs = torch.full(
            (categories,),
            1.0 / categories,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._rates = None
        self.need_update = True

    @abstractmethod
    def inverse_cdf(
        self, parameter: torch.Tensor, quantile: torch.Tensor, invariant: torch.Tensor
    ) -> torch.Tensor:
        ...

    @property
    def invariant(self) -> torch.Tensor:
        return self._invariant.tensor if self._invariant is not None else None

    def update_rates(self, parameter: torch.Tensor, invariant: torch.Tensor):
        if invariant is not None:
            cat = self._categories - 1
            quantile = (2.0 * torch.arange(cat, device=parameter.device) + 1.0) / (
                2.0 * cat
            )
            self.probs = torch.cat(
                (
                    invariant,
                    ((1.0 - invariant) / cat).expand(invariant.shape[:-1] + (cat,)),
                ),
                dim=-1,
            )
            rates = self.inverse_cdf(parameter, quantile, invariant)
        else:
            quantile = (
                2.0 * torch.arange(self._categories, device=parameter.device) + 1.0
            ) / (2.0 * self._categories)
            rates = self.inverse_cdf(parameter, quantile, invariant)

        self._rates = rates / (rates * self.probs).sum(-1, keepdim=True)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self._parameter.tensor, self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self._parameter.tensor, self.invariant)
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


@register_class
class WeibullSiteModel(UnivariateDiscretizedSiteModel):
    @property
    def shape(self) -> torch.Tensor:
        return self._parameter.tensor

    def inverse_cdf(self, parameter, quantile, invariant):
        if invariant is not None:
            return torch.cat(
                (
                    torch.zeros_like(invariant),
                    torch.pow(-torch.log(1.0 - quantile), 1.0 / parameter),
                ),
                dim=-1,
            )
        else:
            return torch.pow(-torch.log(1.0 - quantile), 1.0 / parameter)

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


class LogNormalSiteModel(UnivariateDiscretizedSiteModel):
    @property
    def scale(self) -> torch.Tensor:
        return self._parameter.tensor

    def update_rates(self, value):
        rates = LogNormal(-value * value / 2.0, value).icdf(self.quantile)
        self._rates = rates / (rates.sum() * self.probs)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        scale = process_object(data['scale'], dic)
        categories = int(data['categories'])
        return cls(id_, scale, categories)
