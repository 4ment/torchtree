from __future__ import annotations

import abc
from typing import Any, List

import torch

from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.parameter import Parameter
from torchtree.core.utils import register_class


def set_tensor(parameters, tensor: torch.Tensor) -> None:
    start = 0
    for parameter in parameters:
        parameter.tensor = tensor[
            ..., start : (start + parameter.shape[-1])
        ].requires_grad_()
        start += parameter.shape[-1]


class Integrator(Identifiable, abc.ABC):
    def __init__(self, id_):
        Identifiable.__init__(self, id_)

    @abc.abstractmethod
    def __call__(
        self,
        model: CallableModel,
        parameters: List[Parameter],
        params: torch.Tensor,
        momentum: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def state_dict(self) -> dict[str, Any]:
        state_dict = {"id": self.id}
        state_dict.update(self._state_dict())
        return state_dict

    @abc.abstractmethod
    def _state_dict(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


@register_class
class LeapfrogIntegrator(Integrator):
    def __init__(self, id_, steps: int, step_size: float):
        super().__init__(id_)
        self.steps = steps
        self.step_size = step_size

    def __call__(
        self,
        model: CallableModel,
        parameters: List[Parameter],
        momentum: torch.Tensor,
        inverse_mass_matrix: torch.Tensor,
    ) -> torch.Tensor:
        assert parameters[0].requires_grad is False
        assert momentum.requires_grad is False
        params = torch.cat(
            [parameter.tensor.detach().clone() for parameter in parameters], -1
        )
        momentum = momentum.clone()
        set_tensor(parameters, params)
        U = model()
        U.backward()
        dU = -torch.cat([parameter.grad for parameter in parameters], -1)

        momentum = momentum - self.step_size / 2.0 * dU

        for _ in range(self.steps):
            if inverse_mass_matrix.dim() == 1:
                params = params + self.step_size * inverse_mass_matrix * momentum
            else:
                params = params + self.step_size * (inverse_mass_matrix @ momentum)
            set_tensor(parameters, params.detach())
            U = model()
            U.backward()
            dU = -torch.cat([parameter.grad for parameter in parameters], -1)
            momentum -= self.step_size * dU

        for parameter in parameters:
            parameter.requires_grad = False

        momentum += self.step_size / 2 * dU
        return momentum

    def _state_dict(self) -> dict[str, Any]:
        return {"step_size": self.step_size}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_size = state_dict["step_size"]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        step_size = data.get('step_size', 0.01)
        steps = data.get('steps', 10)
        return cls(id_, steps, step_size)
