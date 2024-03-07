from abc import abstractmethod
from typing import Optional, Union

import torch

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.model import Model
from torchtree.core.parameter import Parameter
from torchtree.core.utils import process_object, process_object_with_key, register_class
from torchtree.typing import ID


class SiteModel(Model):
    _tag = "site_model"

    def __init__(self, id_: ID, mu: AbstractParameter = None) -> None:
        super().__init__(id_)
        self._mu = mu
        self.needs_update = True

    def handle_parameter_changed(self, variable, index, event):
        self.needs_update = True
        self.fire_model_changed()

    def handle_model_changed(self, model, obj, index):
        pass

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
        self.needs_update = False
        return self._rate.tensor

    def probabilities(self) -> torch.Tensor:
        self.needs_update = False
        return self._probability

    def _sample_shape(self) -> torch.Size:
        return self._rate.shape[:-1]

    def to(self, *args, **kwargs) -> None:
        super().to(*args, **kwargs)
        self._probability = self._probability.to(*args, **kwargs)
        self.needs_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda(device)
        self._probability = self._probability.cuda(device)
        self.needs_update = True

    def cpu(self) -> None:
        super().cpu()
        self._probability = self._probability.cpu()
        self.needs_update = True

    @classmethod
    def from_json(cls, data, dic):
        mu = process_object_with_key("mu", data, dic)
        return cls(data["id"], mu)


@register_class
class InvariantSiteModel(SiteModel):
    def __init__(
        self, id_: ID, invariant: AbstractParameter, mu: AbstractParameter = None
    ) -> None:
        super().__init__(id_, mu)
        self._invariant = invariant
        self._rates = None
        self._probabilities = None

    @property
    def invariant(self) -> torch.Tensor:
        return self._invariant.tensor

    def update_rates_probs(self, invariant: torch.Tensor):
        self._probabilities = torch.cat((invariant, 1.0 - invariant), -1)
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
        if self.needs_update:
            self.update_rates_probs(self.invariant)
            self.needs_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.needs_update:
            self.update_rates_probs(self.invariant)
            self.needs_update = False
        return self._probabilities

    def to(self, *args, **kwargs) -> None:
        super().to(*args, **kwargs)
        self._probabilities = self._probabilities.to(*args, **kwargs)
        self.needs_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda(device)
        self._probabilities = self._probabilities.cuda(device)
        self.needs_update = True

    def cpu(self) -> None:
        super().cpu()
        self._probabilities = self._probabilities.cpu()
        self.needs_update = True

    def _sample_shape(self) -> torch.Size:
        return self._invariant.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        invariant = process_object(data["invariant"], dic)
        mu = process_object_with_key("mu", data, dic)
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
        self._probabilities = torch.full(
            (categories,),
            1.0 / categories,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._rates = None
        if invariant is not None:
            self._categories += 1

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
            self._probabilities = torch.cat(
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

        self._rates = rates / (rates * self._probabilities).sum(-1, keepdim=True)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.needs_update:
            self.update_rates(self._parameter.tensor, self.invariant)
            self.needs_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.needs_update:
            self.update_rates(self._parameter.tensor, self.invariant)
            self.needs_update = False
        return self._probabilities

    def _sample_shape(self) -> torch.Size:
        return max(
            [parameter.shape[:-1] for parameter in self._parameters.values()],
            key=len,
        )

    def to(self, *args, **kwargs) -> None:
        super().to(*args, **kwargs)
        self._probabilities = self._probabilities.to(*args, **kwargs)
        self.needs_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda(device)
        self._probabilities = self._probabilities.cuda(device)
        self.needs_update = True

    def cpu(self) -> None:
        super().cpu()
        self._probabilities = self._probabilities.cpu()
        self.needs_update = True


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
        id_ = data["id"]
        categories = data["categories"]
        shape = process_object(data["shape"], dic)
        invariant = process_object_with_key("invariant", data, dic)
        mu = process_object_with_key("mu", data, dic)
        return cls(id_, shape, categories, invariant, mu)
