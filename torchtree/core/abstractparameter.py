import abc
from typing import List, Optional, Union

import torch
from torch import Size, Tensor

from .identifiable import Identifiable


class Device(abc.ABC):
    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        ...

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        ...

    @abc.abstractmethod
    def cpu(self) -> None:
        ...


class AbstractParameter(Identifiable, Device, abc.ABC):
    @property
    @abc.abstractmethod
    def tensor(self) -> Tensor:
        ...

    @tensor.setter
    @abc.abstractmethod
    def tensor(self, tensor: Tensor) -> None:
        ...

    @property
    def shape(self) -> Size:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad

    @requires_grad.setter
    @abc.abstractmethod
    def requires_grad(self, requires_grad: bool) -> None:
        ...

    def dim(self) -> int:
        return self.tensor.dim()

    def parameters(self) -> List['AbstractParameter']:
        return [self]

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @abc.abstractmethod
    def add_parameter_listener(self, listener) -> None:
        ...

    @abc.abstractmethod
    def fire_parameter_changed(self, index=None, event=None) -> None:
        ...

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a.tensor if hasattr(a, 'tensor') else a for a in args]
        return func(*args, **kwargs)
