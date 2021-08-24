from typing import List, Union

import torch
import torch.distributions

from .. import Parameter
from ..core.model import Model
from ..core.utils import process_object, process_objects
from ..distributions.distributions import DistributionModel
from ..typing import ID


class MultivariateNormal(DistributionModel):
    """Multivariate normal distribution.

    :param id_: ID of joint distribution
    :param x: random variable to evaluate/sample using distribution
    :param loc: mean of the distribution
    :param **kwargs: dict keyed with covariance_matrix or precision_matrix or
     scale_tril and valued with Parameter
    :param distributions: list of distributions of type DistributionModel or
     CallableModel
    """

    def __init__(
        self, id_: ID, x: Union[List[Parameter], Parameter], loc: Parameter, **kwargs
    ) -> None:
        super().__init__(id_)
        self.x = x
        self.loc = loc
        self.parameterization = list(kwargs.keys())[0]
        self.parameter = list(kwargs.values())[0]

        self.add_parameter(self.loc)
        self.add_parameter(self.parameter)

        if isinstance(self.x, (list, tuple)):
            for xx in self.x:
                self.add_parameter(xx)
        else:
            self.add_parameter(self.x)

    def _update_tensor(self, x: Union[List[Parameter], Parameter]) -> None:
        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = x[..., offset : (offset + xx.shape[-1])]
                offset += xx.shape[-1]
        else:
            self.x.tensor = x

    def rsample(self, sample_shape=torch.Size()) -> None:
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).rsample(
            sample_shape
        )
        self._update_tensor(x)

    def sample(self, sample_shape=torch.Size()) -> None:
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).sample(
            sample_shape
        )
        self._update_tensor(x)

    def log_prob(self, x: Union[List[Parameter], Parameter] = None) -> torch.Tensor:
        kwargs = {self.parameterization: self.parameter.tensor}
        if isinstance(self.x, (list, tuple)):
            return torch.distributions.MultivariateNormal(
                self.loc.tensor, **kwargs
            ).log_prob(torch.cat([xx.tensor for xx in x], -1))
        else:
            return torch.distributions.MultivariateNormal(
                self.loc.tensor, **kwargs
            ).log_prob(x.tensor)

    def handle_model_changed(self, model: Model, obj, index) -> None:
        pass

    def handle_parameter_changed(self, variable: Parameter, index, event) -> None:
        pass

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob(self.x)

    @property
    def event_shape(self) -> torch.Size:
        return self.loc.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.loc.shape[:-1]

    @property
    def sample_shape(self) -> torch.Size:
        offset = 1 if len(self.batch_shape) == 0 else len(self.batch_shape)
        return self.x.tensor.shape[:-offset]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_objects(data['x'], dic)
        loc = process_object(data['parameters']['loc'], dic)
        kwargs = {}

        for p in ('covariance_matrix', 'scale_tril', 'precision_matrix'):
            if p in data['parameters']:
                parameterization = p
                kwargs[parameterization] = process_object(
                    data['parameters'][parameterization], dic
                )

        if len(kwargs) != 1:
            raise NotImplementedError(
                'MultivariateNormal is parameterized with either covariance_matrix,'
                ' scale_tril or precision_matrix'
            )

        if isinstance(x, list):
            x_count = sum([xx.shape[0] for xx in x])
            assert x_count == loc.shape[0]
            assert torch.Size((x_count, x_count)) == kwargs[parameterization].shape
        else:
            # event_shape must match
            assert x.shape[-1:] == loc.shape[-1:]
            assert x.shape[-1:] + x.shape[-1:] == kwargs[parameterization].shape[-2:]

        return cls(id_, x, loc, **kwargs)
