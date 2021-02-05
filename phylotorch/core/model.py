import abc
import collections.abc
import torch
from phylotorch.utils import process_object, get_class
from phylotorch.serializable import JSONSerializable
import inspect


class Listener(object):
    def __init__(self):
        self.models = []

    def fire(self, model=None, index=None):
        for m in self.models:
            m.handle_update(model, index)


class Model(abc.ABC):
    def __init__(self, id):
        self._id = id

    @property
    def id(self):
        return self._id

    @abc.abstractmethod
    def update(self, value):
        pass

    @abc.abstractmethod
    def handle_update(self, model, index):
        pass


class CallableModel(Model, collections.abc.Callable):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Parameter(Model):
    def __init__(self, id, tensor):
        self._tensor = tensor
        self.listener = Listener()
        super().__init__(id)

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor
        self.listener.fire()

    def assign(self, parameter):
        self._tensor = parameter.tensor
        self.listener.fire()

    def update(self, value):
        if self.id in value:
            self.tensor = value[self.id]

    def handle_update(self, model, index):
        pass

    @classmethod
    def from_json(cls, data, dic):
        id = data['id']
        values = data['tensor']
        t = torch.tensor(values, dtype=torch.float64)
        return cls(id, t)


class TransformedParameter(Parameter):
    example = {
        'id': 'node_heights',
        'type': 'phylotorch.model.TransformedParameter',
        'transform': {
            'id': 'nodeTransform',
            'type': 'TransformModel',
            'transform': 'NodeHeightTransform',
            'parameters':{
                'tree': 'tree'
            }
        },
        'x': ['root_height', 'ratios']
    }

    def __init__(self, id, x, transform):
        self.transform = transform
        self.x = x
        self.need_update = False
        if isinstance(self.x, list):
            tensor = self.transform(torch.cat([x.tensor for x in self.x]))
        else:
            tensor = self.transform(self.x.tensor)
        super().__init__(id, tensor)

    def update(self, value):
        self.x.update(value)
        self._tensor = self.transform(self.x.tensor)

    def handle_update(self, model, index):
        if isinstance(self.x, list):
            self._tensor = self.transform(torch.cat([x.tensor for x in self.x]))
        else:
            self._tensor = self.transform(self.x.tensor)
        self.listener.fire(self)

    def __call__(self):
        if isinstance(self.x, list):
            return self.transform.log_abs_det_jacobian(torch.cat([x.tensor for x in self.x]), self._tensor)
        else:
            return self.transform.log_abs_det_jacobian(self.x.tensor, self._tensor)

    @classmethod
    def from_json(cls, data, dic):
        id = data['id']
        transform = process_object(data['transform'], dic)
        if isinstance(data['x'], list):
            x = []
            for xx in data['x']:
                x.append(process_object(xx, dic))
        else:
            x = process_object(data['x'], dic)
        return cls(id, x, transform)


class TransformModel(Model, JSONSerializable):
    def update(self, value):
        pass

    def handle_update(self, model, index):
        pass

    def __init__(self, id, transform):
        self.transform = transform
        super(TransformModel, self).__init__(id)

    def __call__(self, x):
        return self.transform(x)

    @classmethod
    def from_json(cls, data, dic):
        id = data['id']
        klass = get_class(data['transform'])
        signature_params = list(inspect.signature(klass.__init__).parameters)
        params = []
        for arg in signature_params[1:]:
            if arg in data['parameters']:
                params.append(process_object(data['parameters'][arg], dic))
        transform = klass(*params)
        return cls(id, transform)
