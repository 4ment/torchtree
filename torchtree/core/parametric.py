from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Union

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


class Parametric(ModelListener, ParameterListener, abc.ABC):
    def __init__(self) -> None:
        self._parameters = OrderedDict()
        self._models = OrderedDict()

    def __getattr__(self, name: str) -> Union[AbstractParameter, 'Parametric']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_models' in self.__dict__:
            _models = self.__dict__['_models']
            if name in _models:
                return _models[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __setattr__(
        self, name: str, value: Union[AbstractParameter, 'Parametric']
    ) -> None:
        from .model import Model

        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, AbstractParameter):
            params = self.__dict__.get('_parameters')
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Model.__init__() call"
                )
            remove_from(self.__dict__, self._models)
            self.register_parameter(name, value)
        elif isinstance(value, Model):
            models = self.__dict__.get('_models')
            if models is None:
                raise AttributeError(
                    "cannot assign models before Model.__init__() call"
                )
            remove_from(self.__dict__, self._parameters)
            self.register_model(name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._models:
            del self._models[name]
        else:
            object.__delattr__(self, name)

    def register_parameter(self, name: str, parameter: AbstractParameter) -> None:
        self._parameters[name] = parameter
        parameter.add_parameter_listener(self)

    def register_model(self, name: str, model: 'Parametric') -> None:
        self._models[name] = model
        model.add_model_listener(self)

    def parameters(self) -> list[AbstractParameter]:
        """Returns parameters of instance Parameter."""
        parameters = []
        for param in self._parameters.values():
            parameters.extend(param.parameters())
        for model in self._models.values():
            parameters.extend(model.parameters())
        return parameters
