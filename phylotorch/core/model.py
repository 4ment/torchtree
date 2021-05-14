import abc
import collections.abc
import inspect
import numbers
from typing import List, Union, Optional

import numpy as np
import torch
import torch.distributions
from torch import Tensor
from torch import nn

from .classproperty_decorator import classproperty
from .serializable import JSONSerializable
from ..core.utils import process_object, get_class, tensor_rand


class Identifiable(JSONSerializable):
    def __init__(self, id_: Optional[str]) -> None:
        self._id = id_

    @property
    def id(self) -> Optional[str]:
        return self._id

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data, dic):
        ...


class Parameter(Identifiable):
    def __init__(self, id_: Optional[str], tensor: Tensor) -> None:
        self._tensor = tensor
        self.listeners = []
        super(Parameter, self).__init__(id_)

    def __str__(self):
        return f"{self._id}"

    def __repr__(self):
        return f"Parameter(id_='{self._id}', tensor=torch.{self._tensor})"

    def __eq__(self, other):
        return self.id == other.id and torch.all(torch.eq(self._tensor, other.tensor))

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self._tensor = tensor
        self.fire_parameter_changed()

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._tensor.requires_grad = requires_grad
        self.fire_parameter_changed()

    def assign(self, parameter):
        self._tensor = parameter.tensor
        self.fire_parameter_changed()

    def update(self, value):
        if self.id in value:
            self.tensor = value[self.id]

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    def parameters(self) -> List['Parameter']:
        return [self]

    def clone(self) -> 'Parameter':
        """Return a clone of the Parameter. it is not cloning listeners and the clone's id is None"""
        tclone = self.tensor.clone()
        return Parameter(None, tclone)

    def __getitem__(self, subscript) -> 'Parameter':
        """Can be a slice or index"""
        return Parameter(None, self._tensor[subscript])

    @classmethod
    def from_json(cls, data, dic):
        dtype = get_class(data.get('dtype', 'torch.float64'))

        if 'full_like' in data:
            input_param = process_object(data['full_like'], dic)
            dtype = dtype if 'dtype' in data else input_param.dtype
            if 'rand' in data:
                t = tensor_rand(data['rand'], input_param.dtype, input_param.shape)
            else:
                values = data['tensor']
                t = torch.full_like(input_param.tensor, values, dtype=dtype)
        elif 'full' in data:
            size = data['full']  # a list
            if 'rand' in data:
                t = tensor_rand(data['rand'], dtype, size)
            else:
                values = data['tensor']
                t = torch.full(size, values, dtype=dtype)
        elif 'zeros_like' in data:
            input_param = process_object(data['zeros_like'], dic)
            dtype = dtype if 'dtype' in data else input_param.dtype
            t = torch.zeros_like(input_param.tensor, dtype=dtype)
        elif 'zeros' in data:
            size = data['zeros']
            t = torch.zeros(size, dtype=dtype)
        elif 'ones_like' in data:
            input_param = process_object(data['ones_like'], dic)
            dtype = dtype if 'dtype' in data else input_param.dtype
            t = torch.ones_like(input_param.tensor, dtype=dtype)
        elif 'ones' in data:
            size = data['ones']
            t = torch.ones(size, dtype=dtype)
        elif 'eye' in data:
            size = data['eye']
            t = torch.eye(size, dtype=dtype)
        else:
            values = data['tensor']
            if 'dimension' in data:
                values = np.repeat(values, data['dimension'] / len(values) + 1)
                values = values[:data['dimension']]
            t = torch.tensor(values, dtype=dtype)
        if 'nn' in data and data['nn']:
            return cls(data['id'], nn.Parameter(t))
        else:
            return cls(data['id'], t)


class Parametric(object):
    def __init__(self) -> None:
        self._parameters = []

    def add_parameter(self, parameter: Parameter) -> None:
        self._parameters.append(parameter)

    def parameters(self) -> List[Parameter]:
        """Returns parameters of instance Parameter"""
        parameters = []
        for param in self._parameters:
            parameters.extend(param.parameters())
        return parameters


class ModelListener(abc.ABC):
    @abc.abstractmethod
    def handle_model_changed(self, model, obj, index) -> None:
        ...


class ParameterListener(abc.ABC):
    @abc.abstractmethod
    def handle_parameter_changed(self, variable, index, event) -> None:
        ...


class Model(Identifiable, Parametric, ModelListener, ParameterListener):
    _tag = None

    def __init__(self, id_: Optional[str]) -> None:
        self.listeners = []
        self._models = []
        Identifiable.__init__(self, id_)
        Parametric.__init__(self)

    @abc.abstractmethod
    def update(self, value):
        ...

    def add_model(self, model: 'Model') -> None:
        model.add_model_listener(self)
        self._models.append(model)

    def add_model_listener(self, listener: ModelListener) -> None:
        self.listeners.append(listener)

    def add_parameter(self, parameter: Parameter) -> None:
        parameter.add_parameter_listener(self)
        self._parameters.append(parameter)

    def add_parameter_listener(self, listener: ModelListener) -> None:
        self.listeners.append(listener)

    def fire_model_changed(self, obj=None, index=None) -> None:
        for listener in self.listeners:
            listener.handle_model_changed(self, obj, index)

    def parameters(self) -> List[Parameter]:
        """Returns parameters of instance Parameter"""
        parameters = []
        for param in self._parameters:
            parameters.extend(param.parameters())
        for model in self._models:
            parameters.extend(model.parameters())
        return parameters

    @classproperty
    def tag(cls) -> Optional[str]:
        return cls._tag


class CallableModel(Model, collections.abc.Callable):
    def __init__(self, id_: Optional[str]) -> None:
        Model.__init__(self, id_)
        self.lp = None

    @abc.abstractmethod
    def _call(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        self.lp = self._call(*args, **kwargs)
        return self.lp


class TransformedParameter(Parameter, CallableModel):

    def __init__(self, id_: Optional[str], x: Union[List[Parameter], Parameter],
                 transform: torch.distributions.Transform) -> None:
        CallableModel.__init__(self, id_)
        self.transform = transform
        self.x = x
        self.need_update = False
        if isinstance(self.x, list):
            tensor = self.transform(torch.cat([x.tensor for x in self.x], -1))
            for xx in self.x:
                self.add_parameter(xx)
        else:
            tensor = self.transform(self.x.tensor)
            self.add_parameter(x)
        Parameter.__init__(self, id_, tensor)

    def parameters(self) -> List[Parameter]:
        if isinstance(self.x, list):
            return [param for params in self.x for param in params.parameters()]
        else:
            return self.x.parameters()

    def update(self, value):
        self.x.update(value)
        self._tensor = self.transform(self.x.tensor)

    def _call(self) -> Tensor:
        if self.need_update:
            self.apply_transform()
            self.need_update = False

        if isinstance(self.x, list):
            return self.transform.log_abs_det_jacobian(torch.cat([x.tensor for x in self.x], -1), self._tensor)
        else:
            return self.transform.log_abs_det_jacobian(self.x.tensor, self._tensor)

    @property
    def tensor(self) -> Tensor:
        if self.need_update:
            self.apply_transform()
            self.need_update = False
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        raise Exception('Cannot assign tensor to TransformedParameter (ID: {})'.format(self.id))

    def apply_transform(self) -> None:
        if isinstance(self.x, list):
            self._tensor = self.transform(torch.cat([x.tensor for x in self.x], -1))
        else:
            self._tensor = self.transform(self.x.tensor)

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.need_update = True
        self.fire_parameter_changed()

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    @property
    def batch_shape(self) -> torch.Size:
        # FIXME: is it the shape of self._tensor or self.x
        return self._tensor.shape[-1:]

    @property
    def sample_shape(self) -> torch.Size:
        return self._tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        # parse transform
        klass = get_class(data['transform'])
        signature_params = list(inspect.signature(klass.__init__).parameters)
        params = []
        if 'parameters' in data:
            for arg in signature_params[1:]:
                if arg in data['parameters']:
                    if isinstance(data['parameters'][arg], numbers.Number):
                        params.append(data['parameters'][arg])
                    else:
                        params.append(process_object(data['parameters'][arg], dic))
        transform = klass(*params)

        if isinstance(data['x'], list):
            x = []
            for xx in data['x']:
                x.append(process_object(xx, dic))
        else:
            x = process_object(data['x'], dic)
        return cls(data['id'], x, transform)
