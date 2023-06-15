"""Abstract parameter module."""
import abc
from typing import List, Optional, Union

import torch
from torch import Size, Tensor

from .identifiable import Identifiable


class Device(abc.ABC):
    """Interface for classes needing to allocate a Tensor on a device."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Returns the torch.device where the Tensor is.

        :rtype: torch.device
        """
        ...

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> None:
        """Performs Tensor dtype and/or device conversion."""
        ...

    @abc.abstractmethod
    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        """Moves the tensor object in CUDA memory."""
        ...

    @abc.abstractmethod
    def cpu(self) -> None:
        """Moves the tensor object in CPU memory."""
        ...


class AbstractParameter(Identifiable, Device, abc.ABC):
    """Abstract base class for parameters."""

    @property
    @abc.abstractmethod
    def tensor(self) -> Tensor:
        """The tensor.

        :getter: Returns the tensor.
        :setter: Sets the tensor.
        :rtype: Tensor
        """
        ...

    @tensor.setter
    @abc.abstractmethod
    def tensor(self, tensor: Tensor) -> None:
        ...

    @property
    def shape(self) -> Size:
        """The shape of the tensor.

        :rtype: Size
        """
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the tensor.

        :rtype: torch.dtype
        """
        return self.tensor.dtype

    @property
    def requires_grad(self) -> bool:
        """Is True if gradients need to be computed for this Tensor, False otherwise.

        :getter: Returns the flag.
        :setter: Sets the flag.
        :rtype: bool
        """
        return self.tensor.requires_grad

    @requires_grad.setter
    @abc.abstractmethod
    def requires_grad(self, requires_grad: bool) -> None:
        ...

    def dim(self) -> int:
        """Returns the dimension of the tensor.

        :rtype: int
        """
        return self.tensor.dim()

    def parameters(self) -> List['AbstractParameter']:
        return [self]

    @property
    def device(self) -> torch.device:
        """Returns the torch.device where the Tensor is.

        :rtype: torch.device
        """
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
