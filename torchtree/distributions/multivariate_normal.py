import torch
import torch.distributions

from .. import Parameter
from ..core.abstractparameter import AbstractParameter
from ..core.parameter import CatParameter
from ..core.utils import process_object, process_objects, register_class
from ..distributions.distributions import DistributionModel
from ..typing import ID


@register_class
class MultivariateNormal(DistributionModel):
    """Multivariate normal distribution.

    :param id_: ID of joint distribution
    :param x: random variable to evaluate/sample using distribution
    :param loc: mean of the distribution
    :param covariance_matrix: covariance matrix Parameter
    :param precision_matrixs: precision matrix Parameter
    :param scale_tril: scale tril Parameter
    """

    def __init__(
        self,
        id_: ID,
        x: AbstractParameter,
        loc: AbstractParameter,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
    ) -> None:
        super().__init__(id_)
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or"
                " scale_tril may be specified."
            )
        self.loc = loc
        if covariance_matrix is not None:
            self.parameterization = 'covariance_matrix'
            self.parameter = covariance_matrix
        elif precision_matrix is not None:
            self.parameterization = 'precision_matrix'
            self.parameter = precision_matrix
        else:
            self.parameterization = 'scale_tril'
            self.parameter = scale_tril

        if isinstance(x, (list, tuple)):
            self.x = CatParameter(None, x, dim=-1)
        else:
            self.x = x

    def rsample(self, sample_shape=torch.Size()) -> None:
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).rsample(
            sample_shape
        )
        self.x.tensor = x

    def sample(self, sample_shape=torch.Size()) -> None:
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).sample(
            sample_shape
        )
        self.x.tensor = x

    def log_prob(self, x: Parameter = None) -> torch.Tensor:
        kwargs = {self.parameterization: self.parameter.tensor}
        return torch.distributions.MultivariateNormal(
            self.loc.tensor, **kwargs
        ).log_prob(x.tensor)

    def entropy(self) -> torch.Tensor:
        kwargs = {self.parameterization: self.parameter.tensor}
        return torch.distributions.MultivariateNormal(
            self.loc.tensor, **kwargs
        ).entropy()

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
