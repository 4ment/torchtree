from __future__ import annotations

import abc
import inspect
import numbers
from collections import OrderedDict
from typing import Optional, Type, Union

import torch
import torch.distributions

from .. import Parameter
from ..core.abstractparameter import AbstractParameter
from ..core.container import Container
from ..core.model import CallableModel
from ..core.parameter import CatParameter
from ..core.utils import get_class, process_object, process_objects, register_class


class DistributionModel(CallableModel):
    @abc.abstractmethod
    def rsample(self, sample_shape=torch.Size()) -> None:
        ...

    @abc.abstractmethod
    def sample(self, sample_shape=torch.Size()) -> None:
        ...

    @abc.abstractmethod
    def log_prob(self, x: AbstractParameter = None) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor:
        ...


@register_class
class Distribution(DistributionModel):
    """Wrapper for torch Distribution.

    :param id_: ID of joint distribution
    :param dist: class of torch Distribution
    :param x: random variable to evaluate/sample using distribution
    :param args: parameters of the distribution
    :param **kwargs: optional arguments for instanciating torch Distribution
    """

    def __init__(
        self,
        id_: Optional[str],
        dist: Type[torch.distributions.Distribution],
        x: Union[list[AbstractParameter], AbstractParameter],
        args: 'OrderedDict[str, AbstractParameter]',
        **kwargs,
    ) -> None:
        super().__init__(id_)
        self.dist = dist
        self.args = args
        self.kwargs = kwargs

        self.parameters = Container(None, self.args.values())

        if isinstance(x, (tuple, list)):
            self.x = CatParameter('x', x, dim=-1)
        else:
            self.x = x

    def rsample(self, sample_shape=torch.Size()) -> None:
        x = self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).rsample(sample_shape)
        self.x.tensor = x

    def sample(self, sample_shape=torch.Size()) -> None:
        x = self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).sample(sample_shape)
        self.x.tensor = x

    def log_prob(
        self, x: Union[list[AbstractParameter], AbstractParameter] = None
    ) -> torch.Tensor:
        return self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).log_prob(x.tensor)

    def entropy(self) -> torch.Tensor:
        return self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).entropy()

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob(self.x)

    @property
    def event_shape(self) -> torch.Size:
        return self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return self.dist(
            *[arg.tensor for arg in self.args.values()], **self.kwargs
        ).batch_shape

    @property
    def sample_shape(self) -> torch.Size:
        offset = 1 if len(self.batch_shape) == 0 else len(self.batch_shape)
        return self.x.tensor.shape[:-offset]

    @staticmethod
    def json_factory(
        id_: str,
        distribution: str,
        x: Union[str, dict],
        parameters: Union[str, dict] = None,
    ) -> dict:
        distr = {
            'id': id_,
            'type': 'Distribution',
            'distribution': distribution,
            'x': x,
        }
        if parameters is not None:
            distr['parameters'] = parameters
        return distr

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        klass = get_class(data['distribution'])
        x = process_objects(data['x'], dic)

        signature_params = list(inspect.signature(klass.__init__).parameters)
        params = OrderedDict()
        if 'parameters' not in data:
            return cls(id_, klass, x, params)

        data_dist = data['parameters']
        for arg in signature_params[1:]:
            if arg in data_dist:
                if isinstance(data_dist[arg], str):
                    params[arg] = dic[data_dist[arg]]
                elif isinstance(data_dist[arg], numbers.Number):
                    params[arg] = Parameter(
                        None, torch.tensor(data_dist[arg], dtype=x.dtype)
                    )
                elif isinstance(data_dist[arg], list):
                    params[arg] = Parameter(
                        None, torch.tensor(data_dist[arg], dtype=x.dtype)
                    )
                else:
                    params[arg] = process_object(data_dist[arg], dic)

        return cls(id_, klass, x, params)
