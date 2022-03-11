import abc
from typing import Iterator, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module

from .abstractparameter import AbstractParameter


class ModelListener(abc.ABC):
    @abc.abstractmethod
    def handle_model_changed(self, model, obj, index) -> None:
        ...


class ParameterListener(abc.ABC):
    @abc.abstractmethod
    def handle_parameter_changed(
        self, variable: AbstractParameter, index, event
    ) -> None:
        ...


class Parametric(Module, ModelListener, ParameterListener, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.listeners = []

    def register_parameter(self, name: str, param) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        from .. import Parameter

        # print(name)
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(
                "parameter name should be a string. "
                "Got {}".format(torch.typename(name))
            )
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' "
                "(torch.nn.Parameter or None required)".format(
                    torch.typename(param), name
                )
            )
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name)
            )
        else:
            self._parameters[name] = param
            param.add_parameter_listener(self)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        super().add_module(name, module)
        if name is not None and isinstance(module, Parametric):
            self._modules[name].add_model_listener(self)

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        from .. import Parameter

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        else:
            super().__setattr__(name, value)
            if isinstance(value, Parametric):
                self._modules[name].add_model_listener(self)

    def parameters(
        self, recurse: bool = True, tensor=False
    ) -> Union[Iterator[AbstractParameter], List[torch.nn.Parameter]]:
        """Returns parameters of instance Parameter."""
        parameters = []
        if tensor:
            for param in super().parameters(recurse):
                yield param
        else:
            for param in self._parameters.values():
                parameters.extend(list(param.parameters()))
                print(self.id, param.id, param.shape)
            for model in self._modules.values():
                parameters.extend(list(model.parameters()))

        return parameters

    def add_model_listener(self, listener: ModelListener) -> None:
        self.listeners.append(listener)

    def remove_model_listener(self, listener: ModelListener) -> None:
        self.listeners.remove(listener)

    def add_parameter_listener(self, listener: ParameterListener) -> None:
        self.listeners.append(listener)

    def remove_parameter_listener(self, listener: ParameterListener) -> None:
        self.listeners.remove(listener)

    def fire_model_changed(self, obj=None, index=None) -> None:
        for listener in self.listeners:
            listener.handle_model_changed(self, obj, index)
