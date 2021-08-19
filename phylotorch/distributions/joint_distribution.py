from typing import List, Union

import torch.distributions

from .. import Parameter
from ..core.model import Model
from ..core.utils import process_object
from ..typing import ID
from .distributions import DistributionModel


class JointDistributionModel(DistributionModel):
    """Joint distribution of independent distributions.

    :param id_: ID of joint distribution
    :param distributions: list of distributions of type DistributionModel or
     CallableModel
    """

    def __init__(self, id_: ID, distributions: List[DistributionModel]) -> None:
        super().__init__(id_)
        self.distributions = distributions
        for distr in self.distributions:
            self.add_model(distr)

    def log_prob(self, x: Union[List[Parameter], Parameter] = None) -> torch.Tensor:
        log_p = []
        for distr in self.distributions:
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
            else:
                log_p.append(lp)
        return torch.cat(log_p, -1).sum(-1)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob()

    def rsample(self, sample_shape=torch.Size()) -> None:
        for distr in self.distributions:
            distr.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()) -> None:
        for distr in self.distributions:
            distr.sample(sample_shape)

    def update(self, value):
        pass

    def handle_model_changed(self, model: Model, obj, index) -> None:
        self.fire_model_changed()

    def handle_parameter_changed(self, variable: Parameter, index, event) -> None:
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return max([model.sample_shape for model in self._models], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        distributions = []
        for d in data['distributions']:
            distributions.append(process_object(d, dic))
        return cls(id_, distributions)
