import abc
import collections.abc
import inspect

import numpy as np
import torch

from ..core.utils import process_object, get_class
from .serializable import JSONSerializable


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
        super(Model, self).__init__(id_)

    @abc.abstractmethod
    def update(self, value):
        pass

    def add_model(self, model):
        model.add_model_listener(self)

    def add_model_listener(self, listener):
        self.listeners.append(listener)

    def add_parameter(self, parameter):
        parameter.add_parameter_listener(self)

    def add_parameter_listener(self, listener):
        self.listeners.append(listener)

    def fire_model_changed(self, obj=None, index=None):
        for listener in self.listeners:
            listener.handle_model_changed(self, obj, index)


class CallableModel(Model, collections.abc.Callable):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Parameter(Identifiable):
    def __init__(self, id_, tensor):
        self._tensor = tensor
        self.listeners = []
        super(Parameter, self).__init__(id_)

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

    @classmethod
    def from_json(cls, data, dic):
        values = data['tensor']
        if 'dimension' in data:
            values = np.repeat(values, data['dimension'])
        t = torch.tensor(values, dtype=torch.float64)
        return cls(data['id'], t)


class TransformedParameter(Parameter, ParameterListener):

    def __init__(self, id_, x, transform):
        self.transform = transform
        self.x = x
        self.need_update = False
        if isinstance(self.x, list):
            tensor = self.transform(torch.cat([x.tensor for x in self.x]))
            for xx in self.x:
                xx.add_parameter_listener(self)
        else:
            tensor = self.transform(self.x.tensor)
            x.add_parameter_listener(self)
        super().__init__(id_, tensor)

    def update(self, value):
        self.x.update(value)
        self._tensor = self.transform(self.x.tensor)

    def __call__(self):
        if isinstance(self.x, list):
            return self.transform.log_abs_det_jacobian(torch.cat([x.tensor for x in self.x]), self._tensor)
        else:
            return self.transform.log_abs_det_jacobian(self.x.tensor, self._tensor)

    def handle_parameter_changed(self, variable, index, event):
        if isinstance(self.x, list):
            self._tensor = self.transform(torch.cat([x.tensor for x in self.x]))
        else:
            self._tensor = self.transform(self.x.tensor)
        self.fire_parameter_changed()

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
                    params.append(process_object(data['parameters'][arg], dic))
        transform = klass(*params)
        return cls(data['id'], transform)
