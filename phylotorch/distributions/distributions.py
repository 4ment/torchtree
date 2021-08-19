import abc
import inspect
import numbers
from collections import OrderedDict
from typing import List, Optional, Type, Union

import torch
import torch.distributions

from .. import Parameter
from ..core.model import CallableModel, Model
from ..core.utils import get_class, process_object, process_objects


class DistributionModel(CallableModel):
    @abc.abstractmethod
    def rsample(self, sample_shape=torch.Size()) -> None:
        ...

    @abc.abstractmethod
    def sample(self, sample_shape=torch.Size()) -> None:
        ...

    @abc.abstractmethod
    def log_prob(self, x: Union[List[Parameter], Parameter] = None) -> torch.Tensor:
        ...


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
        x: Union[List[Parameter], Parameter],
        args: 'OrderedDict[str, Parameter]',
        **kwargs
    ) -> None:
        super().__init__(id_)
        self.dist = dist
        self.x = x
        self.args = args
        self.kwargs = kwargs

        for p in self.args.values():
            if p.id is not None:
                self.add_parameter(p)
        if isinstance(self.x, list):
            for xx in self.x:
                self.add_parameter(xx)
        else:
            self.add_parameter(self.x)

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

    def log_prob(self, x: Union[List[Parameter], Parameter] = None) -> torch.Tensor:
        if isinstance(self.x, list):
            return self.dist(
                *[arg.tensor for arg in self.args.values()], **self.kwargs
            ).log_prob(torch.cat([xx.tensor for xx in x], -1))
        else:
            return self.dist(
                *[arg.tensor for arg in self.args.values()], **self.kwargs
            ).log_prob(x.tensor)

    def update(self, value):
        for name in self.args.keys():
            if name in value:
                self.args[name] = value[name]

    def handle_model_changed(self, model: Model, obj, index) -> None:
        pass

    def handle_parameter_changed(self, variable: Parameter, index, event) -> None:
        pass

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
