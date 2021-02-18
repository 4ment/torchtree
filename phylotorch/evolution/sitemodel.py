from abc import abstractmethod

import torch
from torch.distributions import LogNormal

from ..core.model import Model
from ..core.utils import process_object


class SiteModel(Model):

    def __init__(self, id_):
        super(SiteModel, self).__init__(id_)

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

    def __init__(self, id_):
        super(ConstantSiteModel, self).__init__(id_)

    def update(self, value):
        pass

    def rates(self):
        return torch.tensor([1.0], dtype=torch.float64)

    def probabilities(self):
        return torch.tensor([1.0], dtype=torch.float64)

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        return cls(data['id'])


class InvariantSiteModel(SiteModel):

    def __init__(self, id_, invariant):
        self._invariant = invariant
        self._rates = None
        self._probs = None
        self.need_update = True
        self.add_parameter(self._invariant)
        super(InvariantSiteModel, self).__init__(id_)

    @property
    def invariant(self):
        return self._invariant.tensor

    def update_rates_probs(self, invariant):
        self._probs = torch.cat((invariant, 1.0 - invariant))
        self._rates = torch.cat((torch.zeros_like(invariant), 1.0 / (1.0 - invariant)))

    def update(self, value):
        if isinstance(value, dict):
            if self._invariant.id in value:
                self._invariant.tensor = value[self._invariant.id]
                self.update_rates_probs(self.invariant)
        else:
            self.update_rates_probs(value)

    def rates(self):
        if self.need_update:
            self.update_rates_probs(self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self):
        if self.need_update:
            self.update_rates_probs(self.invariant)
            self.need_update = False
        return self._probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        invariant = process_object(data['invariant'], dic)
        return cls(id_, invariant)


class WeibullSiteModel(SiteModel):

    def __init__(self, id_, shape, categories, invariant=None):
        self._shape = shape
        self.categories = categories
        self._invariant = invariant
        self.probs = torch.full((categories,), 1.0 / categories, dtype=self.shape.dtype)
        self._rates = None
        self.need_update = True
        self.add_parameter(self._shape)
        super(WeibullSiteModel, self).__init__(id_)

    @property
    def shape(self):
        return self._shape.tensor

    @property
    def invariant(self):
        return self._invariant.tensor if self._invariant else None

    def update_rates(self, shape, invariant):
        if invariant:
            cat = self.categories - 1
            quantile = (2.0 * torch.arange(cat) + 1.0) / (2.0 * cat)
            self.probs = torch.cat((invariant, torch.full((cat,), (1.0 - invariant) / cat)))
            rates = torch.cat((torch.zeros_like(invariant), torch.pow(-torch.log(1.0 - quantile), 1.0 / shape)))
        else:
            quantile = (2.0 * torch.arange(self.categories) + 1.0) / (2.0 * self.categories)
            rates = torch.pow(-torch.log(1.0 - quantile), 1.0 / shape)

        self._rates = rates / (rates * self.probs).sum()

    def update(self, value):
        if isinstance(value, dict):
            update = False
            if self._shape.id in value:
                update = True
                self._shape.tensor = value[self._shape.id]
            if self._invariant and self._invariant.id in value:
                update = True
                self._invariant.tensor = value[self._invariant.id]
            if update:
                self.update_rates(self.shape, self.invariant)
        else:
            self.update_rates(value, None)

    def rates(self):
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self):
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self.probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        shape = process_object(data['shape'], dic)
        categories = data['categories']
        invariant = None
        if 'invariant' in data:
            invariant = process_object(data['invariant'], dic)
        return cls(id_, shape, categories, invariant)


class LogNormalSiteModel(SiteModel):

    def __init__(self, id_, scale, categories=4):
        self._scale = scale
        self.categories = categories
        self.probs = torch.full((categories,), 1.0 / categories, dtype=self.scale.dtype)
        self.quantile = (2.0 * torch.arange(categories) + 1.0) / (2.0 * categories)
        self._rates = None
        self.update_rates(self.scale)
        self.need_update = True
        self.add_parameter(self._scale)
        super(LogNormalSiteModel, self).__init__(id_)

    @property
    def scale(self):
        return self._scale.tensor

    def update_rates(self, value):
        rates = LogNormal(-value * value / 2., value).icdf(self.quantile)
        self._rates = rates / (rates.sum() * self.probs)

    def update(self, value):
        if isinstance(value, dict):
            if self._scale.id in value:
                self._scale.tensor = value[self._scale.id]
                self.update_rates(self.scale)
        else:
            self.update_rates(value)

    def rates(self):
        if self.need_update:
            self.update_rates(self.scale)
            self.need_update = False
        return self._rates

    def probabilities(self):
        return self.probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        scale = process_object(data['scale'], dic)
        categories = int(data['categories'])
        return cls(id_, scale, categories)
