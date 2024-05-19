from __future__ import annotations

import abc
import math
import statistics
from collections import deque
from typing import Any

import torch
from torch import Tensor

from torchtree.core.identifiable import Identifiable
from torchtree.core.utils import process_objects, register_class
from torchtree.typing import ID, Parameter


class MCMCOperator(Identifiable, abc.ABC):
    def __init__(
        self,
        id_: ID,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        **kwargs,
    ):
        super().__init__(id_)
        self.parameters = parameters
        self.weight = weight
        self.target_acceptance_probability = target_acceptance_probability
        self._adapt_count = 0
        self._accept = 0
        self._reject = 0
        self._disable_adaptation = kwargs.get("disable_adaptation", False)
        self._accept_window_length = kwargs.get("acceptance_window_length", 100)
        self._accept_window = deque()

    @property
    @abc.abstractmethod
    def tuning_parameter(self) -> float:
        pass

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
        self._accept_window.append(1)
        if len(self._accept_window) > self._accept_window_length:
            self._accept_window.popleft()

    def reject(self) -> None:
        for parameter, saved_tensor in zip(self.parameters, self.saved_tensors):
            parameter.tensor = saved_tensor
        self._reject += 1
        self._accept_window.append(0)
        if len(self._accept_window) > self._accept_window_length:
            self._accept_window.popleft()

    def smoothed_acceptance_rate(self) -> float:
        if len(self._accept_window) == 0:
            return math.nan
        return statistics.mean(self._accept_window)

    def tune(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        if not self._disable_adaptation:
            assert 0.0 <= acceptance_prob <= 1.0
            new_parameter = self.adaptable_parameter + (
                acceptance_prob.item() - self.target_acceptance_probability
            ) / (2 + self._adapt_count)
            self.adaptable_parameter = new_parameter

    def state_dict(self) -> dict[str, Any]:
        state_dict = {
            "id": self.id,
            "adapt_count": self._adapt_count,
            "accept": self._accept,
            "reject": self._reject,
            "accept_window": list(self._accept_window),
        }
        state_dict.update(self._state_dict())
        return state_dict

    @abc.abstractmethod
    def _state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._adapt_count = state_dict["adapt_count"]
        self._accept = state_dict["accept"]
        self._reject = state_dict["reject"]
        self._accept_window = deque(state_dict["accept_window"])
        self._load_state_dict(state_dict)

    @abc.abstractmethod
    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    @staticmethod
    def _parse_json(data, dic):
        id_ = data["id"]
        parameters = process_objects(data["parameters"], dic, force_list=True)
        weight = data.get("weight", 1.0)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.24)
        optionals = {}
        optionals["disable_adaptation"] = data.get("disable_adaptation", False)
        optionals["acceptance_window_length"] = data.get(
            "acceptance_window_length", False
        )
        return id_, parameters, weight, target_acceptance_probability, optionals


@register_class
class ScalerOperator(MCMCOperator):
    def __init__(
        self,
        id_: ID,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        scaler: float,
        **kwargs,
    ):
        super().__init__(
            id_, parameters, weight, target_acceptance_probability, **kwargs
        )
        self._scaler = scaler

    @property
    def tuning_parameter(self) -> float:
        return self._scaler

    @MCMCOperator.adaptable_parameter.getter
    def adaptable_parameter(self) -> float:
        return math.log(1.0 / self._scaler - 1.0)

    def set_adaptable_parameter(self, value: float) -> None:
        self._scaler = 1.0 / (math.exp(value) + 1.0)

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

    def _state_dict(self) -> dict[str, Any]:
        return {"scaler": self._scaler}

    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler = state_dict["scaler"]

    @classmethod
    def from_json(cls, data, dic):
        (
            id_,
            parameters,
            weight,
            target_acceptance_probability,
            optionals,
        ) = MCMCOperator._parse_json(data, dic)
        scaler = data.get("scaler", 0.1)

        return cls(
            id_, parameters, weight, target_acceptance_probability, scaler, **optionals
        )


@register_class
class SlidingWindowOperator(MCMCOperator):
    def __init__(
        self,
        id_: ID,
        parameters: list[Parameter],
        weight: float,
        target_acceptance_probability: float,
        width: float,
        **kwargs,
    ) -> None:
        super().__init__(
            id_, parameters, weight, target_acceptance_probability, **kwargs
        )
        self._width = width

    @property
    def tuning_parameter(self) -> float:
        return self._width

    @MCMCOperator.adaptable_parameter.getter
    def adaptable_parameter(self) -> float:
        return math.log(self._width)

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

    def _state_dict(self) -> dict[str, Any]:
        return {"width": self._width}

    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._width = state_dict["width"]

    @classmethod
    def from_json(cls, data, dic):
        (
            id_,
            parameters,
            weight,
            target_acceptance_probability,
            optionals,
        ) = MCMCOperator._parse_json(data, dic)
        width = data.get("width", 0.1)

        return cls(
            id_, parameters, weight, target_acceptance_probability, width, **optionals
        )


@register_class
class DirichletOperator(MCMCOperator):
    def __init__(
        self,
        id_: ID,
        parameters: Parameter,
        weight: float,
        target_acceptance_probability: float,
        scaler: float,
        **kwargs,
    ) -> None:
        super().__init__(
            id_, parameters, weight, target_acceptance_probability, **kwargs
        )
        self._scaler = scaler

    @property
    def tuning_parameter(self) -> float:
        return self._scaler

    @MCMCOperator.adaptable_parameter.getter
    def adaptable_parameter(self) -> float:
        return math.log(self._scaler)

    def set_adaptable_parameter(self, value: float) -> None:
        self._scaler = math.exp(value)

    def _step(self) -> Tensor:
        old_values = self.parameters[0].tensor
        scaled_old = old_values * self._scaler
        dist_old = torch.distributions.Dirichlet(scaled_old)
        new_values = dist_old.sample()
        scaled_new = new_values * self._scaler
        self.parameters[0].tensor = new_values

        f = dist_old.log_prob(new_values)
        b = torch.distributions.Dirichlet(scaled_new).log_prob(old_values)

        return b - f

    def _state_dict(self) -> dict[str, Any]:
        return {"scaler": self._scaler}

    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler = state_dict["scaler"]

    @classmethod
    def from_json(cls, data, dic):
        (
            id_,
            parameters,
            weight,
            target_acceptance_probability,
            optionals,
        ) = MCMCOperator._parse_json(data, dic)
        scaler = data.get("scaler", 1.0)

        return cls(
            id_, parameters, weight, target_acceptance_probability, scaler, **optionals
        )
