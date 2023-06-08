from __future__ import annotations

import abc
import math

import torch
from torch import Tensor

from torchtree.core.utils import process_objects, register_class

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
        self._disable_adaptation = kwargs.get("disable_adaptation", False)

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
        if not self._disable_adaptation:
            assert 0.0 <= acceptance_prob <= 1.0
            new_parameter = self.adaptable_parameter + (
                acceptance_prob - self.target_acceptance_probability
            ) / (2 + self._adapt_count)
            self.adaptable_parameter = new_parameter


@register_class
class ScalerOperator(MCMCOperator):
    def __init__(
        self,
        id_: ID,
        joint: CallableModel,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        scaler: float,
        **kwargs,
    ):
        super().__init__(
            id_, joint, parameters, weight, target_acceptance_probability, **kwargs
        )
        self._scaler = scaler

    @property
    def adaptable_parameter(self) -> float:
        return math.log(self._scaler)

    @adaptable_parameter.setter
    def adaptable_parameter(self, value: float) -> None:
        self.set_adaptable_parameter(value)
        self._adapt_count += 1

    def set_adaptable_parameter(self, value: float) -> None:
        self._scaler = math.exp(value)

    def _step(self) -> Tensor:
        s = self._scaler + (
            torch.rand(1).item() * ((1.0 / self._scaler) - self._scaler)
        )
        index = torch.randint(0, len(self.parameters), (1,)).item()
        index2 = torch.randint(0, len(self.parameters[index].tensor), (1,)).item()
        p = self.parameters[index].tensor
        p[index2] *= s
        self.parameters[index].tensor = p
        # this does not trigger listeners:
        # self.parameters[index].tensor[index2] *= s

        return -torch.tensor(
            s, device=self.parameters[0].device, dtype=self.parameters[0].dtype
        ).log()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        joint = process_objects(data["joint"], dic)
        parameters = process_objects(data["parameters"], dic, force_list=True)
        weight = data.get("weight", 1.0)
        scaler = data.get("scaler", 0.1)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.24)

        return cls(
            id_,
            joint,
            parameters,
            weight,
            target_acceptance_probability,
            scaler,
        )


@register_class
class SlidingWindowOperator(MCMCOperator):
    def __init__(
        self,
        id_: ID,
        joint: CallableModel,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        width: float,
        **kwargs,
    ) -> None:
        super().__init__(
            id_, joint, parameters, weight, target_acceptance_probability, **kwargs
        )
        self._width = width

    @property
    def adaptable_parameter(self) -> float:
        return math.log(self._width)

    @adaptable_parameter.setter
    def adaptable_parameter(self, value: float) -> None:
        self.set_adaptable_parameter(value)
        self._adapt_count += 1

    def set_adaptable_parameter(self, value: float) -> None:
        self._width = math.exp(value)

    def _step(self) -> Tensor:
        shift = self._width * (torch.rand(1).item() - 0.5)
        index = torch.randint(0, len(self.parameters), (1,)).item()
        index2 = torch.randint(0, len(self.parameters[index].tensor), (1,)).item()
        p = self.parameters[index].tensor
        p[index2] += shift
        self.parameters[index].tensor = p

        return torch.tensor(
            0.0, device=self.parameters[0].device, dtype=self.parameters[0].dtype
        )

    def tune(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        if not self._disable_adaptation:
            super().tune(acceptance_prob, sample, accepted)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        joint = process_objects(data["joint"], dic)
        parameters = process_objects(data["parameters"], dic, force_list=True)
        weight = data.get("weight", 1.0)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.24)
        width = data.get("width", 0.1)

        return cls(
            id_,
            joint,
            parameters,
            weight,
            target_acceptance_probability,
            width,
        )
