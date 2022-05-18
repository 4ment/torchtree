from __future__ import annotations

from typing import Optional, Union

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import Model
from ..core.parameter import CatParameter
from ..core.utils import process_objects, register_class
from ..distributions.distributions import DistributionModel


@register_class
class DeterministicNormal(DistributionModel):
    """Deterministic Normal distribution.

    Standard normal variates are drawn during object creation and samples drawn
    from this distribution are a transformation of these variates

    :param id_: ID of joint distribution
    :param x: random variable to evaluate/sample using distribution
    :param loc: location of the distribution
    :param scale: scale of the distribution
    :param shape: shape of standard normal variates
    """

    def __init__(
        self,
        id_: Optional[str],
        loc: AbstractParameter,
        scale: AbstractParameter,
        x: Union[list[AbstractParameter], AbstractParameter],
        shape: torch.Size,
    ) -> None:
        super().__init__(id_)
        self.loc = loc
        self.scale = scale

        if isinstance(x, (tuple, list)):
            self.x = CatParameter('x', x, dim=-1)
        else:
            self.x = x
        self.eps = torch.empty(shape + self.loc.shape).normal_()

    def rsample(self, sample_shape=torch.Size()) -> None:
        self.x.tensor = self.loc.tensor + self.eps * self.scale.tensor

    def sample(self, sample_shape=torch.Size()) -> None:
        with torch.no_grad():
            self.x.tensor = self.loc.tensor + self.eps * self.scale.tensor

    def log_prob(
        self, x: Union[list[AbstractParameter], AbstractParameter] = None
    ) -> torch.Tensor:
        return torch.distributions.Normal(self.loc.tensor, self.scale.tensor).log_prob(
            x.tensor
        )

    def entropy(self) -> torch.Tensor:
        return torch.distributions.Normal(self.loc.tensor, self.scale.tensor).entropy()

    def handle_model_changed(self, model: Model, obj, index) -> None:
        pass

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob(self.x)

    @property
    def event_shape(self) -> torch.Size:
        return torch.distributions.Normal(
            self.loc.tensor, self.scale.tensor
        ).event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return torch.distributions.Normal(
            self.loc.tensor, self.scale.tensor
        ).batch_shape

    @property
    def sample_shape(self) -> torch.Size:
        offset = 1 if len(self.batch_shape) == 0 else len(self.batch_shape)
        return self.x.tensor.shape[:-offset]

    @staticmethod
    def json_factory(
        id_: str,
        loc: Union[str, dict],
        scale: Union[str, dict],
        x: Union[str, dict],
        shape: list,
    ) -> dict:
        distr = {
            'id': id_,
            'type': 'DeterministicNormal',
            'loc': loc,
            'scale': scale,
            'shape': shape,
            'x': x,
        }
        return distr

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        if 'parameters' in data:
            loc = process_objects(data['parameters']['loc'], dic)
            scale = process_objects(data['parameters']['scale'], dic)
        else:
            loc = process_objects(data['loc'], dic)
            scale = process_objects(data['scale'], dic)
        x = process_objects(data['x'], dic)
        shape = torch.Size(data['shape'])
        return cls(id_, loc, scale, x, shape)
