from __future__ import annotations

import abc

from torch import Tensor

from ...core.identifiable import Identifiable
from ...core.model import CallableModel
from ...typing import ID, Parameter


class MCMCOperator(Identifiable, abc.ABC):
    def __init__(
        self,
        id_: ID,
        joint: CallableModel,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        **kwargs,
    ):
        super().__init__(id_)
        self._joint = joint
        self.parameters = parameters
        self.weight = weight
        self.target_acceptance_probability = target_acceptance_probability
        self._adapt_count = 0
        self._accept = 0
        self._reject = 0

    @property
    @abc.abstractmethod
    def adaptable_parameter(self) -> float:
        pass

    @adaptable_parameter.setter
    def adaptable_parameter(self, value: float) -> None:
        self.set_adaptable_parameter(value)
        self._adapt_count += 1

    @abc.abstractmethod
    def set_adaptable_parameter(self, value: float) -> None:
        pass

    @abc.abstractmethod
    def _step(self) -> Tensor:
        pass

    def step(self) -> Tensor:
        self.saved_tensors = [parameter.tensor.clone() for parameter in self.parameters]
        return self._step()

    def accept(self) -> None:
        self._accept += 1

    def reject(self) -> None:
        for parameter, saved_tensor in zip(self.parameters, self.saved_tensors):
            parameter.tensor = saved_tensor
        self._reject += 1

    def tune(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        assert 0.0 <= acceptance_prob <= 1.0
        new_parameter = self.adaptable_parameter + (
            acceptance_prob - self.target_acceptance_probability
        ) / (2 + self._adapt_count)
        self.adaptable_parameter = new_parameter
