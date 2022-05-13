from __future__ import annotations

import torch
from torch import Tensor

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.parameter import Parameter
from ..core.utils import process_object
from ..typing import ID


class RootParameter(AbstractParameter, CallableModel):
    r"""This root height parameter is calculated from
     number of substitutions / substitution rate.

    :param id_: ID of object
    :type id_: str or None
    :param Parameter distance: number of substitution parameter
    :param Parameter rate: rate parameter
    :param float shift: shift root height by this amount. Used by serially sampled trees
    """

    def __init__(
        self, id_: ID, distance: Parameter, rate: Parameter, shift: float
    ) -> None:
        CallableModel.__init__(self, id_)
        AbstractParameter.__init__(self, id_)
        self.distance = distance
        self.rate = rate
        self.shift = shift
        self.need_update = False
        self._tensor = self.transform()
        self.listeners = []

    def parameters(self) -> list[Parameter]:
        return [self.distance, self.rate]

    def _call(self) -> Tensor:
        if self.need_update:
            self._tensor = self.transform()
            self.need_update = False

        return -self.rate.tensor.log()

    @property
    def tensor(self) -> Tensor:
        if self.need_update:
            self._tensor = self.transform()
            self.need_update = False
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        raise Exception(
            'Cannot assign tensor to TransformedParameter (ID: {})'.format(self.id)
        )

    def transform(self) -> Tensor:
        """Return root height."""
        return self.distance.tensor / self.rate.tensor + self.shift

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.need_update = True
        super().handle_parameter_changed(variable, index, event)

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    def add_parameter_listener(self, listener) -> None:
        self.listeners.append(listener)

    def fire_parameter_changed(self, index=None, event=None) -> None:
        for listener in self.listeners:
            listener.handle_parameter_changed(self, index, event)

    @property
    def sample_shape(self) -> torch.Size:
        return self._tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic) -> RootParameter:
        r"""
        Create a RootParameter object.

        :param data: json representation of RootParameter object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
         in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.evolution.root_transform.RootParameter` object.
        :rtype: RootParameter
        """
        x: Parameter = process_object(data['x'], dic)
        rate: Parameter = process_object(data['rate'], dic)
        shift = data.get('shift', 0.0)
        return cls(data['id'], x, rate, shift)
