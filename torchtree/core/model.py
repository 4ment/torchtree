import abc
import collections.abc
from typing import Optional, Union

import torch.distributions
from torch import Tensor

from torchtree.core.abstractparameter import AbstractParameter

from .classproperty_decorator import classproperty
from .identifiable import Identifiable
from .parametric import ModelListener, ParameterListener, Parametric


class Model(Parametric, Identifiable, ModelListener, ParameterListener):
    _tag = None

    def __init__(self, id_: Optional[str]) -> None:
        Parametric.__init__(self)
        Identifiable.__init__(self, id_)
        self.listeners = []

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

    @classproperty
    def tag(cls) -> Optional[str]:
        return cls._tag

    def to(self, *args, **kwargs) -> None:
        self._apply(lambda x: x.to(*args, **kwargs))

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._apply(lambda x: x.cuda(device))

    def cpu(self) -> None:
        self._apply(lambda x: x.cpu())

    def _apply(self, fn):
        for param in self._parameters:
            fn(param)
        for model in self._models:
            fn(model)

    def models(self):
        for model in self._models.values():
            yield model

    @property
    @abc.abstractmethod
    def sample_shape(self) -> torch.Size:
        ...


class CallableModel(Model, collections.abc.Callable):
    """Classes extending CallableModel are Callable and the returned value is
    cached in case we need to use this value multiple times without the need to
    recompute it."""

    def __init__(self, id_: Optional[str]) -> None:
        Model.__init__(self, id_)
        self.lp = None
        self.lp_needs_update = True

    @abc.abstractmethod
    def _call(self, *args, **kwargs) -> Tensor:
        pass

    def handle_parameter_changed(
        self, variable: AbstractParameter, index, event
    ) -> None:
        self.lp_needs_update = True
        self.fire_model_changed(self)

    def handle_model_changed(self, model, obj, index) -> None:
        self.lp_needs_update = True
        self.fire_model_changed(self)

    def __call__(self, *args, **kwargs) -> Tensor:
        if self.lp_needs_update:
            self.lp = self._call(*args, **kwargs)
            self.lp_needs_update = False
        return self.lp
