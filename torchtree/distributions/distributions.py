"""torchtree distribution classes."""
from __future__ import annotations

import abc
import inspect
import numbers
from typing import Any, Optional, Type, Union

import torch
import torch.distributions

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.container import Container
from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.parameter import CatParameter, Parameter
from torchtree.core.utils import (
    get_class,
    process_object,
    process_objects,
    register_class,
)


class DistributionModel(CallableModel):
    """Abstract base class for distribution models."""

    @abc.abstractmethod
    def rsample(self, sample_shape=torch.Size()) -> None:
        """Generates a sample_shape shaped reparameterized sample or
        sample_shape shaped batch of reparameterized samples if the
        distribution parameters are batched."""
        ...

    @abc.abstractmethod
    def sample(self, sample_shape=torch.Size()) -> None:
        """Generates a sample_shape shaped sample or sample_shape shaped batch
        of samples if the distribution parameters are batched."""
        ...

    @abc.abstractmethod
    def log_prob(self, x: AbstractParameter = None) -> torch.Tensor:
        """Returns the log of the probability density/mass function evaluated
        at x.

        :param Parameter x: value to evaluate
        :return: log probability
        :rtype: Tensor
        """
        ...

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor:
        """Returns entropy of distribution, batched over batch_shape.

        :return: Tensor of shape batch_shape.
        :rtype: Tensor
        """
        ...


@register_class
class Distribution(DistributionModel):
    """Wrapper for :class:`torch.distributions.distribution.Distribution`.

    :param id_: ID of distribution
    :param dist: class of torch Distribution
    :param x: random variable to evaluate/sample using distribution
    :param dict[str, AbstractParameter] parameters: parameters of the distribution
    :param **kwargs: optional arguments for instanciating torch Distribution
    """

    def __init__(
        self,
        id_: Optional[str],
        dist: Type[torch.distributions.Distribution],
        x: Union[list[AbstractParameter], AbstractParameter],
        parameters: dict[str, AbstractParameter],
        **kwargs,
    ) -> None:
        super().__init__(id_)
        self.dist = dist
        self.dict_parameters = parameters
        self.kwargs = kwargs

        # this is for listening to changes.
        self.distribution_parameters = Container(
            None, list(self.dict_parameters.values())
        )

        if isinstance(x, (tuple, list)):
            self.x = CatParameter('x', x, dim=-1)
        else:
            self.x = x

    def rsample(self, sample_shape=torch.Size()) -> None:
        x = self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).rsample(sample_shape)
        self.x.tensor = x

    def sample(self, sample_shape=torch.Size()) -> None:
        x = self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).sample(sample_shape)
        self.x.tensor = x

    def log_prob(
        self, x: Union[list[AbstractParameter], AbstractParameter] = None
    ) -> torch.Tensor:
        return self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).log_prob(x.tensor)

    def entropy(self) -> torch.Tensor:
        return self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).entropy()

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return self.log_prob(self.x)

    @property
    def event_shape(self) -> torch.Size:
        return self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        ).batch_shape

    def _sample_shape(self) -> torch.Size:
        offset = 1 if len(self.batch_shape) == 0 else len(self.batch_shape)
        return self.x.tensor.shape[:-offset]

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return self.dist(
            **{name: p.tensor for name, p in self.dict_parameters.items()},
            **self.kwargs,
        )

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
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> Distribution:
        r"""Creates a Distribution object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a
            Distribution object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - distribution (str): complete path to the torch distribution class,
            including package and module.
          - x (dict or str): parameter.

         Optional:
          - parameters (dict): parameters of the underlying torch Distribution.

        **JSON examples**:

        .. code-block:: json

          {
            "id": "exp",
            "distribution": "torch.distributions.Exponential",
            "x": {
                "id": "y",
                "type": "Parameter",
                "tensor": 0.1
            },
            "parameters": {
              "rate": {
                "id": "rate",
                "type": "Parameter",
                "tensor": 0.1
              }
            }
          }

        .. code-block:: json

          {
            "id": "normal",
            "distribution": "torch.distributions.Normal",
            "x": {
                "id": "y",
                "type": "Parameter",
                "tensor": 0.1
            },
            "parameters": {
              "loc": {
                "id": "loc",
                "type": "Parameter",
                "tensor": 0.0
              },
              "scale": {
                "id": "scale",
                "type": "Parameter",
                "tensor": 0.1
              }
            }
          }

        :example:
        >>> x_dict = {"id": "x", "type": "Parameter", "tensor": [1., 2.]}
        >>> x = Parameter.from_json(x_dict, {})
        >>> dic = {"x": x}
        >>> loc = {"id": "loc", "type": "Parameter", "tensor": [0.1]}
        >>> scale = {"id": "scale", "type": "Parameter", "tensor": [1.]}
        >>> normal_dic = {"id": "normal", "distribution": "torch.distributions.Normal",
        ...     "x": "x", "parameters":{"loc": loc, "scale": scale}}
        >>> normal = Distribution.from_json(normal_dic, dic)
        >>> isinstance(normal, Distribution)
        True
        >>> exp_dic = {"id": "exp", "x": "x", "parameters":{"rate": 1.0},
        ...     "distribution": "torch.distributions.Exponential"}
        >>> exp = Distribution.from_json(exp_dic, dic)
        >>> exp() == torch.distributions.Exponential(1.0).log_prob(x.tensor)
        tensor([True, True])

        .. note::
            The names of the keys in the `parameters` dictionary must match the
            variable names used in the signature of the torch distributions.
            See https://pytorch.org/docs/stable/distributions.html.
        """
        id_ = data['id']
        klass = get_class(data['distribution'])
        x = process_objects(data['x'], dic)

        signature_params = list(inspect.signature(klass.__init__).parameters)
        params = {}
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
