from __future__ import annotations

import collections.abc
import inspect
import numbers
from typing import Optional, Union, overload

import numpy as np
import torch
from torch import Tensor, nn

from .abstractparameter import AbstractParameter
from .container import Container
from .parametric import ParameterListener, Parametric
from .utils import (
    JSONParseError,
    get_class,
    process_object,
    process_objects,
    register_class,
    tensor_rand,
)


@register_class
class Parameter(AbstractParameter):
    def __init__(self, id_: Optional[str], tensor: Tensor) -> None:
        super().__init__(id_)
        self._tensor = tensor
        self.listeners = []

    def __str__(self):
        return f"{self._id}"

    def __repr__(self):
        id_ = "'" + self._id + "'" if self._id else None
        return f"Parameter(id_={id_}, tensor=torch.{self._tensor})"

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
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._tensor.requires_grad = requires_grad
        self.fire_parameter_changed()

    def detach(self):
        return Parameter(self.id, self._tensor.detach())

    @property
    def grad_fn(self):
        return self._tensor.grad_fn

    def copy_(self, tensor):
        self._tensor.copy_(tensor)

    def size(self):
        return self._tensor.size()

    @property
    def grad(self) -> bool:
        return self._tensor.grad

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    def clone(self) -> Parameter:
        """Return a clone of the Parameter.

        it is not cloning listeners and the clone's id is None
        """
        tclone = self.tensor.clone()
        return Parameter(None, tclone)

    def __getitem__(self, subscript) -> Parameter:
        """Can be a slice, int, or Tensor."""
        return Parameter(None, self._tensor[subscript])

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._tensor = self._tensor.cuda(device)

    def cpu(self) -> None:
        self._tensor = self._tensor.cpu()

    @overload
    def to(
        self,
        device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ) -> None:
        ...

    @overload
    def to(self, dtype: Union[torch.dtype, str] = None) -> None:
        ...

    def to(self, *args, **kwargs) -> None:
        """Performs Tensor dtype and/or device conversion.

        A torch.dtype and torch.device are inferred from the arguments
        of self.to(*args, **kwargs)

        This can be called as

        .. function:: to(device=None, dtype=None)

        .. function:: to(dtype)

        .. function:: to(device)
        """
        if len(args) == 0:
            self._tensor = self._tensor.to(
                device=kwargs['device'], dtype=kwargs['dtype']
            )
        else:
            self._tensor = self._tensor.to(args[0])

    @staticmethod
    def json_factory(id_: str, **kwargs):
        parameter = {
            'id': id_,
            'type': 'Parameter',
        }
        if 'full_like' in kwargs:
            parameter['full_like'] = kwargs['full_like']
            parameter['tensor'] = kwargs['tensor']
        elif 'full' in kwargs:
            parameter['full'] = kwargs['full']
            parameter['tensor'] = kwargs['tensor']
        elif 'zeros_like' in kwargs:
            parameter['zeros_like'] = kwargs['zeros_like']
        elif 'zeros' in kwargs:
            parameter['zeros'] = kwargs['zeros']
        elif 'ones_like' in kwargs:
            parameter['ones_like'] = kwargs['ones_like']
        elif 'ones' in kwargs:
            parameter['ones'] = kwargs['ones']
        elif 'eye' in kwargs:
            parameter['eye'] = kwargs['eye']
        elif 'tensor' in kwargs:
            parameter['tensor'] = kwargs['tensor']

        if 'dtype' in kwargs:
            parameter['dtype'] = kwargs['dtype']
        if 'device' in kwargs:
            parameter['device'] = kwargs['device']
        return parameter

    @classmethod
    def from_json(cls, data, dic):
        dtype = get_class(data['dtype']) if 'dtype' in data else None
        device = data.get('device', None)
        kwargs = {'device': device}
        kwargs['requires_grad'] = data.get('requires_grad', False)

        if 'full_like' in data:
            input_param = process_object(data['full_like'], dic)
            if 'rand' in data:
                t = tensor_rand(data['rand'], input_param.shape, **kwargs)
            else:
                values = data['tensor']
                t = torch.full_like(input_param.tensor, values, **kwargs)
        elif 'full' in data:
            if dtype:
                kwargs['dtype'] = dtype
            size = data['full']  # a list
            if 'rand' in data:
                t = tensor_rand(data['rand'], size, **kwargs)
            else:
                values = data['tensor']
                t = torch.full(size, values, **kwargs)
        elif 'zeros_like' in data:
            input_param = process_object(data['zeros_like'], dic)
            t = torch.zeros_like(input_param.tensor, **kwargs)
        elif 'zeros' in data:
            if dtype:
                kwargs['dtype'] = dtype
            size = data['zeros']
            t = torch.zeros(size, **kwargs)
        elif 'ones_like' in data:
            input_param = process_object(data['ones_like'], dic)
            t = torch.ones_like(input_param.tensor, **kwargs)
        elif 'ones' in data:
            if dtype:
                kwargs['dtype'] = dtype
            size = data['ones']
            t = torch.ones(size, **kwargs)
        elif 'eye' in data:
            if dtype:
                kwargs['dtype'] = dtype
            size = data['eye']
            t = torch.eye(size, **kwargs)
        else:
            if dtype:
                kwargs['dtype'] = dtype
            values = data['tensor']
            if 'dimension' in data:
                values = np.repeat(values, data['dimension'] / len(values) + 1)
                values = values[: data['dimension']].tolist()
            t = torch.tensor(values, **kwargs)
        if 'nn' in data and data['nn']:
            return cls(data['id'], nn.Parameter(t))
        return cls(data['id'], t)


@register_class
class TransformedParameter(AbstractParameter, Parametric, collections.abc.Callable):
    def __init__(
        self,
        id_: Optional[str],
        x: Union[list[AbstractParameter], AbstractParameter],
        transform: torch.distributions.Transform,
    ) -> None:
        Parametric.__init__(self)
        AbstractParameter.__init__(self, id_)
        self.transform = transform
        self.need_update = False
        if isinstance(x, list):
            self.x = CatParameter(None, x, -1)
        else:
            self.x = x
        self._tensor = self.transform(self.x.tensor)
        self.listeners = []

    def parameters(self) -> list[AbstractParameter]:
        return self.x.parameters()

    def __call__(self, *args, **kwargs) -> Tensor:
        if self.need_update:
            self.apply_transform()
            self.need_update = False

        return self.transform.log_abs_det_jacobian(self.x.tensor, self._tensor)

    @property
    def tensor(self) -> Tensor:
        if self.need_update:
            self.apply_transform()
            self.need_update = False
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self.x.tensor = self.transform.inv(tensor)

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self.x.requires_grad = requires_grad

    @property
    def shape(self) -> torch.Size:
        if self.need_update:
            self.apply_transform()
            self.need_update = False
        return self._tensor.shape

    def apply_transform(self) -> None:
        self._tensor = self.transform(self.x.tensor)

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.need_update = True
        self.fire_parameter_changed()

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    @property
    def sample_shape(self) -> torch.Size:
        return self.tensor.shape[:-1]

    def to(self, *args, **kwargs) -> None:
        for param in self._parameters:
            param.to(*args, **kwargs)
        self.need_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        for param in self._parameters:
            param.cuda(device)
        self.need_update = True

    def cpu(self):
        for param in self._parameters:
            param.cpu()
        self.need_update = True

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
                    elif isinstance(data['parameters'][arg], list):
                        params.append(
                            Parameter(None, torch.tensor(data['parameters'][arg]))
                        )
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


@register_class
class ViewParameter(AbstractParameter, ParameterListener):
    def __init__(
        self,
        id_: Optional[str],
        parameter: Parameter,
        indices: Union[int, slice, Tensor],
    ) -> None:
        r"""
        Class representing a view of another parameter.

        :param id_: ID of object
        :type id_: str or None
        :param Parameter parameter: parameter that ViewParameter wrap
        :param indices: indices used on parameter
        """
        AbstractParameter.__init__(self, id_)
        self.parameter = parameter
        self.indices = indices
        self.listeners = []
        self.parameter.add_parameter_listener(self)

    def __str__(self):
        return f"{self._id}"

    def __repr__(self):
        if isinstance(self.indices, slice):
            indices = []
            for x in (self.indices.start, self.indices.stop, self.indices.step):
                if x is None:
                    indices.append('')
                else:
                    indices.append(x)
            indices = "'{}:{}:{}'".format(*indices)
        elif isinstance(self.indices, torch.Tensor):
            indices = self.indices.tolist()
        else:
            indices = self.indices
        return (
            f"ViewParameter(id_='{self._id}', parameter={repr(self.parameter)},"
            f" indices={indices})"
        )

    def __eq__(self, other):
        return (
            self.id == other.id
            and self.parameter.__eq__(other.parameter)
            and self.indices.__eq__(other.indices)
        )

    @property
    def tensor(self) -> Tensor:
        return self.parameter.tensor[..., self.indices]

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self.parameter.tensor[..., self.indices] = tensor
        self.parameter.fire_parameter_changed()

    @property
    def shape(self) -> torch.Size:
        return self.parameter.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.parameter.dtype

    @property
    def requires_grad(self) -> bool:
        return self.parameter.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        raise Exception('requires_grad setter cannot be called on ViewParameter')

    def assign(self, parameter):
        raise Exception('assign cannot be called on ViewParameter')

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    def clone(self) -> ViewParameter:
        """Return a clone of the Parameter.

        it is not cloning listeners and the clone's id is None
        """
        return ViewParameter(None, self.parameter.clone(), self.indices)

    def __getitem__(self, subscript) -> ViewParameter:
        raise NotImplementedError

    def handle_parameter_changed(self, variable, index, event) -> None:
        # propagate event when self.parameter changes
        self.fire_parameter_changed()

    def to(self, *args, **kwargs) -> None:
        self.parameter.to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self.parameter.cuda(device)

    def cpu(self) -> None:
        self.parameter.cpu()

    @staticmethod
    def json_factory(id_: str, x, indices):
        return {'id': id_, 'type': 'ViewParameter', 'parameter': x, 'indices': indices}

    @classmethod
    def from_json(cls, data, dic):
        parameter = process_object(data['parameter'], dic)
        indices = None
        if isinstance(data['indices'], int):
            # we 'lose' a dimension in this case
            # example: torch.tensor([0,1,2])[1] == torch.tensor(1)
            # use slicing if we want to keep the original shape
            # example: torch.tensor([0,1,2])[1:2] == torch.tensor([1])
            indices = data['indices']
        elif isinstance(data['indices'], list):
            if isinstance(data['indices'], int):
                indices = torch.LongTensor(data['indices'])
            elif isinstance(data['indices'], bool):
                indices = torch.BoolTensor(data['indices'])
        elif isinstance(data['indices'], str):
            # [ <first element to include> : <first element to exclude> : <step> ]
            slice_indexes = data['indices'].split(':')
            for idx, value in enumerate(slice_indexes):
                if value != '':
                    slice_indexes[idx] = int(value)
                else:
                    slice_indexes[idx] = None

            if len(slice_indexes) == 3 and slice_indexes[2] < 0:
                if slice_indexes[0] is None:
                    start = parameter.tensor.size(-1) - 1
                else:
                    start = slice_indexes[0]
                if slice_indexes[1] is None:
                    end = -1
                else:
                    end = slice_indexes[1]
                indices = torch.arange(start, end, slice_indexes[2])
            else:
                indices = slice(*slice_indexes)
        elif isinstance(data['indices'], Parameter):
            tensor = data['indices'].tensor
            if isinstance(tensor, (torch.FloatTensor, torch.BoolTensor)):
                indices = tensor

        if indices is None:
            raise JSONParseError(
                'indices must be a string, list, or Parameter'
                ' (with a FloatTensor or BoolTensor)'
            )

        return cls(data['id'], parameter, indices)


@register_class
class CatParameter(AbstractParameter, ParameterListener):
    """Class for concatenating parameters.

    :param id_: ID of object
    :param parameters: list or tuple of parameters
    :param dim: dimension for concatenation
    """

    def __init__(
        self,
        id_: Optional[str],
        parameters: Union[list[Parameter], tuple[Parameter, ...]],
        dim: Optional[int] = 0,
    ) -> None:
        AbstractParameter.__init__(self, id_)
        self._parameter_container = Container(None, parameters)
        self._dim = dim
        self._tensor = None
        self._need_update = True
        self.update()
        for parameter in self._parameter_container.params():
            parameter.add_parameter_listener(self)
        self._listeners = []

    def __str__(self):
        return f"{self._id}"

    def __repr__(self):
        id_ = "'" + self._id + "'" if self._id else None
        return (
            f"CatParameter(id_={id_},"
            f" parameters={repr(list(self._parameter_container.params()))},"
            f" dim={self._dim})"
        )

    def __eq__(self, other):
        return (
            self.id == other.id
            and sum(
                [
                    a.__eq__(b)
                    for a, b in zip(
                        self._parameter_container.params(),
                        other._parameter_container.params(),
                    )
                ]
            )
            == len(list(self._parameter_container.params()))
            and self._dim.__eq__(other._dim)
        )

    def update(self):
        if self._need_update:
            self._tensor = torch.cat(
                [parameter.tensor for parameter in self._parameter_container.params()],
                dim=self._dim,
            )
            self._need_update = False

    @property
    def tensor(self) -> Tensor:
        self.update()
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        start = 0
        for parameter in self._parameter_container.params():
            parameter.tensor = tensor[..., start : (start + parameter.shape[-1])]
            start += parameter.shape[-1]
        self._need_update = True

    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        for parameter in self._parameter_container.params():
            parameter.requires_grad = requires_grad
        self._need_update = True

    def to(self, *args, **kwargs) -> None:
        for parameter in self._parameter_container.params():
            parameter.to(*args, **kwargs)
        self._need_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        for parameter in self._parameter_container.params():
            parameter.cuda(device)
        self._need_update = True

    def cpu(self) -> None:
        for parameter in self._parameter_container.params():
            parameter.cpu()
        self._need_update = True

    @property
    def device(self) -> torch.device:
        return next(self._parameter_container.params()).device

    def add_parameter_listener(self, listener) -> None:
        self._listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self._listeners:
            listener.handle_parameter_changed(self, index, event)

    def handle_model_changed(self, variable, index, event) -> None:
        # parameters are inside a Container, which is a model
        self._need_update = True
        self.fire_parameter_changed()

    def handle_parameter_changed(self, variable, index, event) -> None:
        self._need_update = True
        self.fire_parameter_changed()

    @classmethod
    def from_json(cls, data, dic):
        parameters = process_objects(data['parameters'], dic)
        dim = data.get('dim', 0)
        return cls(data['id'], parameters, dim)


@register_class
class ModuleParameter(AbstractParameter, Parametric):
    def __init__(
        self,
        id_: Optional[str],
        module,
    ) -> None:
        Parametric.__init__(self)
        AbstractParameter.__init__(self, id_)
        self.need_update = False
        self.module = module
        self._tensor = module()
        self.listeners = []

    def parameters(self) -> list[AbstractParameter]:
        return self.module.parameters()

    @property
    def tensor(self) -> Tensor:
        if self.need_update:
            self._tensor = self.module()
            self.need_update = False
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self.x.tensor = self.transform.inv(tensor)

    @property
    def requires_grad(self) -> bool:
        raise Exception('requires_grad getter cannot be called on ModuleParameter')

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        raise Exception('requires_grad setter cannot be called on ModuleParameter')

    @property
    def shape(self) -> torch.Size:
        if self.need_update:
            self._tensor = self.module()
            self.need_update = False
        return self._tensor.shape

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.need_update = True
        self.fire_parameter_changed()

    def handle_model_changed(self, model, obj, index) -> None:
        self.need_update = True
        self.fire_parameter_changed()

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    @property
    def sample_shape(self) -> torch.Size:
        if self.need_update:
            self._tensor = self.module()
            self.need_update = False
        return self._tensor.shape[:-1]

    def to(self, *args, **kwargs) -> None:
        self.module.to(*args, **kwargs)
        self.need_update = True

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self.module.cuda(device)
        self.need_update = True

    def cpu(self):
        self.module.cpu()
        self.need_update = True

    @classmethod
    def from_json(cls, data, dic):
        module = process_object(data['module'], dic)
        return cls(data['id'], module)
