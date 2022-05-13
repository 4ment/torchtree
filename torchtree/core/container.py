from __future__ import annotations

import collections.abc
from typing import Optional, Union

from torch import Size

from .abstractparameter import AbstractParameter
from .model import Model


class Container(Model):
    """Container for multiple objects of type Model or AbstractParameter.

    This class inherits from Model so an object referencing this object should be
    listening for model updates (class inherits from ModelListener).

    :param id_: ID of objects
    :param objects: list of Models or AbstractParameters
    """

    def __init__(
        self, id_: Optional[str], objects: list[Union['Model', AbstractParameter]]
    ):
        super().__init__(id_)
        for obj in objects:
            setattr(self, self._unique_id(obj), obj)

    def _unique_id(self, obj):
        """parameters and objects should have unique IDs but one of them can
        have an ID that clashes with other attributes."""
        index = 0
        id_ = str(obj.id)
        unique_id = id_
        while hasattr(self, unique_id):
            index += 1
            unique_id = id_ + str(index)
        return unique_id

    def params(self):
        for parameter in self._parameters.values():
            yield parameter

    def callables(self):
        for model in self.models():
            yield model
        for parameter in self.params():
            if isinstance(parameter, collections.abc.Callable):
                yield parameter

    @property
    def sample_shape(self) -> Size:
        sample_models = Size([])
        sample_parameters = Size([])
        if len(self._parameters) > 0:
            sample_parameters = max(
                [parameter.shape[:-1] for parameter in self._parameters.values()],
                key=len,
            )
        if len(self._models) > 0:
            sample_models = max(
                [model.sample_shape for model in self._models.values()], key=len
            )
        return max(sample_models, sample_parameters, key=len)

    def handle_model_changed(self, model, obj, index) -> None:
        self.fire_model_changed(self)

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.fire_model_changed(self)

    @classmethod
    def from_json(cls, data, dic):
        raise NotImplementedError
