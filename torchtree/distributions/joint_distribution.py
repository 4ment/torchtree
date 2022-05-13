from __future__ import annotations

from typing import Union

import torch.distributions

from .. import Parameter
from ..core.container import Container
from ..core.utils import process_object, register_class
from ..typing import ID
from .distributions import DistributionModel


@register_class
class JointDistributionModel(DistributionModel):
    """Joint distribution of independent distributions.

    :param id_: ID of joint distribution
    :param distributions: list of distributions of type DistributionModel or
     CallableModel
    """

    def __init__(self, id_: ID, distributions: list[DistributionModel]) -> None:
        super().__init__(id_)
        self._distributions = Container(None, distributions)

    def log_prob(self, x: Union[list[Parameter], Parameter] = None) -> torch.Tensor:
        log_p = []
        for distr in self._distributions.callables():
            lp = distr()
            sample_shape = distr.sample_shape
            if lp.shape == sample_shape:
                log_p.append(lp.unsqueeze(-1))
            elif lp.shape == torch.Size([]):
                log_p.append(lp.unsqueeze(0))
            elif lp.shape[-1] != 1:
                log_p.append(lp.sum(-1, keepdim=True))
            elif lp.dim() == 1:
                log_p.append(lp.expand(self.sample_shape + (1,)))
            elif lp.dim() > 1 and len(self.sample_shape) == 0:
                log_p.append(lp.squeeze(0))
            else:
                log_p.append(lp)
        return torch.cat(log_p, -1).sum(-1)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob()

    def rsample(self, sample_shape=torch.Size()) -> None:
        for distr in self._distributions.models():
            distr.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()) -> None:
        for distr in self._distributions.models():
            distr.sample(sample_shape)

    def entropy(self) -> torch.Tensor:
        entropies = []
        for distr in self._distributions.models():
            entropies.append(distr.entropy())
        return torch.cat(entropies, 0).sum()

    def handle_parameter_changed(self, variable: Parameter, index, event) -> None:
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self._distributions.sample_shape

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        distributions = []
        for d in data['distributions']:
            distributions.append(process_object(d, dic))
        return cls(id_, distributions)
