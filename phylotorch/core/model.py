import abc
import collections.abc
import inspect
import numbers
from typing import List, Optional, Union, overload

import numpy as np
import torch
import torch.distributions
from torch import Tensor, nn

from ..core.utils import JSONParseError, get_class, process_object, tensor_rand
from .classproperty_decorator import classproperty
from .serializable import JSONSerializable


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


class AbstractParameter(Identifiable, abc.ABC):
    @property
    @abc.abstractmethod
    def tensor(self) -> Tensor:
        ...

    @tensor.setter
    @abc.abstractmethod
    def tensor(self, tensor: Tensor) -> None:
        ...

    @property
    @abc.abstractmethod
    def shape(self) -> torch.Size:
        ...

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        ...

    @property
    @abc.abstractmethod
    def requires_grad(self) -> bool:
        ...

    @requires_grad.setter
    @abc.abstractmethod
    def requires_grad(self, requires_grad: bool) -> None:
        ...

    @abc.abstractmethod
    def dim(self) -> int:
        ...

    @abc.abstractmethod
    def parameters(self) -> List['AbstractParameter']:
        ...

    @abc.abstractmethod
    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        ...

    @abc.abstractmethod
    def cpu(self) -> None:
        ...

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def add_parameter_listener(self, listener) -> None:
        ...

    @abc.abstractmethod
    def fire_parameter_changed(self, index=None, event=None) -> None:
        ...


class Parameter(AbstractParameter):
    def __init__(self, id_: Optional[str], tensor: Tensor) -> None:
        super().__init__(id_)
        self._tensor = tensor
        self.listeners = []

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

    def dim(self) -> int:
        return self._tensor.dim()

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
        """Return a clone of the Parameter.

        it is not cloning listeners and the clone's id is None
        """
        tclone = self.tensor.clone()
        return Parameter(None, tclone)

    def __getitem__(self, subscript) -> 'Parameter':
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
        if 'dtype' in kwargs:
            self._tensor = self._tensor.to(args[0], dtype=kwargs['dtype'])
        else:
            self._tensor = self._tensor.to(args[0])

    @property
    def device(self) -> torch.device:
        return self._tensor.device

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
                values = values[: data['dimension']]
            t = torch.tensor(values, dtype=dtype)
        if 'nn' in data and data['nn']:
            return cls(data['id'], nn.Parameter(t))
        return cls(data['id'], t)


class Parametric:
    def __init__(self) -> None:
        self._parameters = []

    def add_parameter(self, parameter: Parameter) -> None:
        self._parameters.append(parameter)
        parameter.add_parameter_listener(self)

    def parameters(self) -> List[Parameter]:
        """Returns parameters of instance Parameter."""
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
        """Returns parameters of instance Parameter."""
        parameters = []
        for param in self._parameters:
            parameters.extend(param.parameters())
        for model in self._models:
            parameters.extend(model.parameters())
        return parameters

    @classproperty
    def tag(cls) -> Optional[str]:
        return cls._tag

    def to(self, *args, **kwargs) -> None:
        for param in self._parameters:
            param.to(*args, **kwargs)
        for model in self._models:
            model.cuda(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        for param in self._parameters:
            param.cuda(device)
        for model in self._models:
            model.cuda(device)

    def cpu(self) -> None:
        for param in self._parameters:
            param.cpu()
        for model in self._models:
            model.cpu()

    @property
    @abc.abstractmethod
    def sample_shape(self) -> torch.Size:
        ...


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
    def __init__(
        self,
        id_: Optional[str],
        x: Union[List[Parameter], Parameter],
        transform: torch.distributions.Transform,
    ) -> None:
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

    def _call(self, *args, **kwargs) -> Tensor:
        if self.need_update:
            self.apply_transform()
            self.need_update = False

        if isinstance(self.x, list):
            return self.transform.log_abs_det_jacobian(
                torch.cat([x.tensor for x in self.x], -1), self._tensor
            )
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
        raise Exception(
            'Cannot assign tensor to TransformedParameter (ID: {})'.format(self.id)
        )

    @property
    def shape(self) -> torch.Size:
        if self.need_update:
            self.apply_transform()
            self.need_update = False
        return self._tensor.shape

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
        super().__init__(id_)
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

    def dim(self) -> int:
        return self.tensor.dim()

    def assign(self, parameter):
        raise Exception('assign cannot be called on ViewParameter')

    def update(self, value):
        if self.id in value:
            self.tensor = value[self.id]

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    def parameters(self) -> List['ViewParameter']:
        return [self]

    def clone(self) -> 'ViewParameter':
        """Return a clone of the Parameter.

        it is not cloning listeners and the clone's id is None
        """
        return ViewParameter(None, self.parameter.clone(), self.indices)

    def __getitem__(self, subscript) -> 'ViewParameter':
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
