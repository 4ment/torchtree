import abc
import collections.abc
from typing import Optional, Union

import torch.distributions
from torch import Tensor

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

    def add_parameter_listener(self, listener: ParameterListener) -> None:
        self.listeners.append(listener)

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
    def __init__(self, id_: Optional[str]) -> None:
        Model.__init__(self, id_)
        self.lp = None

    @abc.abstractmethod
    def _call(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        self.lp = self._call(*args, **kwargs)
        return self.lp
