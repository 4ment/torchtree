import abc
import collections.abc
import inspect
import numbers

import numpy as np
import torch

from .serializable import JSONSerializable
from ..core.utils import process_object, get_class


class Identifiable(JSONSerializable):
    def __init__(self, id_):
        self._id = id_

    @property
    def id(self):
        return self._id

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data, dic):
        ...


class ModelListener(abc.ABC):
    @abc.abstractmethod
    def handle_model_changed(self, model, obj, index):
        ...


class ParameterListener(abc.ABC):
    @abc.abstractmethod
    def handle_parameter_changed(self, variable, index, event):
        ...


class Model(Identifiable, ModelListener, ParameterListener):
    def __init__(self, id_):
        self.listeners = []
        self._parameters = []
        self._models = []
        super(Model, self).__init__(id_)

    @abc.abstractmethod
    def update(self, value):
        pass

    def add_model(self, model):
        model.add_model_listener(self)
        self._models.append(model)

    def add_model_listener(self, listener):
        self.listeners.append(listener)

    def add_parameter(self, parameter):
        parameter.add_parameter_listener(self)
        self._parameters.append(parameter)

    def add_parameter_listener(self, listener):
        self.listeners.append(listener)

    def fire_model_changed(self, obj=None, index=None):
        for listener in self.listeners:
            listener.handle_model_changed(self, obj, index)

    def parameters(self):
        """Returns parameters of instance Parameter"""
        parameters = []
        for param in self._parameters:
            parameters.extend(param.parameters())
        for model in self._models:
            parameters.extend(model.parameters())
        return parameters


class CallableModel(Model, collections.abc.Callable):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Parameter(Identifiable):
    def __init__(self, id_, tensor):
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
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor
        self.fire_parameter_changed()

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def requires_grad(self):
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self._tensor.requires_grad = requires_grad
        self.fire_parameter_changed()

    def assign(self, parameter):
        self._tensor = parameter.tensor
        self.fire_parameter_changed()

    def update(self, value):
        if self.id in value:
            self.tensor = value[self.id]

    def add_parameter_listener(self, listener):
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None):
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    def parameters(self):
        return [self]

    @classmethod
    def from_json(cls, data, dic):
        dtype = get_class(data.get('dtype', 'torch.float64'))

        if 'full_like' in data:
            input_param = process_object(data['full_like'], dic)
            dtype = dtype if 'dtype' in data else input_param.dtype
            values = data['tensor']
            t = torch.full_like(input_param.tensor, values, dtype=dtype)
        elif 'full' in data:
            size = data['full']  # a list
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
        else:
            values = data['tensor']
            if 'dimension' in data:
                values = np.repeat(values, data['dimension'] / len(values) + 1)
                values = values[:data['dimension']]
            t = torch.tensor(values, dtype=dtype)
        return cls(data['id'], t)


class TransformedParameter(Parameter, CallableModel):

    def __init__(self, id_, x, transform):
        CallableModel.__init__(self, id_)
        self.transform = transform
        self.x = x
        self.need_update = False
        if isinstance(self.x, list):
            tensor = self.transform(torch.cat([x.tensor for x in self.x]))
            for xx in self.x:
                self.add_parameter(xx)
        else:
            tensor = self.transform(self.x.tensor)
            self.add_parameter(x)
        Parameter.__init__(self, id_, tensor)

    def parameters(self):
        if isinstance(self.x, list):
            return [param for params in self.x for param in params.parameters()]
        else:
            return self.x.parameters()

    def update(self, value):
        self.x.update(value)
        self._tensor = self.transform(self.x.tensor)

    def __call__(self):
        if self.need_update:
            self.apply_transform()
            self.need_update = False

        if isinstance(self.x, list):
            return self.transform.log_abs_det_jacobian(torch.cat([x.tensor for x in self.x]), self._tensor)
        else:
            return self.transform.log_abs_det_jacobian(self.x.tensor, self._tensor)

    @property
    def tensor(self):
        if self.need_update:
            self.apply_transform()
            self.need_update = False
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        raise Exception('Cannot assign tensor to TransformedParameter (ID: {})'.format(self.id))

    def apply_transform(self):
        if isinstance(self.x, list):
            self._tensor = self.transform(torch.cat([x.tensor for x in self.x]))
        else:
            self._tensor = self.transform(self.x.tensor)

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_parameter_changed()

    def handle_model_changed(self, model, obj, index):
        pass

    @classmethod
    def from_json(cls, data, dic):
        transform = process_object(data['transform'], dic)
        if isinstance(data['x'], list):
            x = []
            for xx in data['x']:
                x.append(process_object(xx, dic))
        else:
            x = process_object(data['x'], dic)
        return cls(data['id'], x, transform)


class TransformModel(Model):

    def __init__(self, id_, transform):
        self.transform = transform
        super(TransformModel, self).__init__(id_)

    def __call__(self, x):
        return self.transform(x)

    def log_abs_det_jacobian(self, x, y):
        return self.transform.log_abs_det_jacobian(x, y)

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def update(self, value):
        pass

    @classmethod
    def from_json(cls, data, dic):
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
        return cls(data['id'], transform)
