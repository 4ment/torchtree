import abc
from typing import List

import torch

from torchtree.core.model import CallableModel
from torchtree.core.parameter import Parameter

from ...core.serializable import JSONSerializable
from ...core.utils import register_class


def set_tensor(parameters, tensor: torch.Tensor) -> None:
    start = 0
    for parameter in parameters:
        parameter.tensor = tensor[
            ..., start : (start + parameter.shape[-1])
        ].requires_grad_()
        start += parameter.shape[-1]


class Integrator(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        model: CallableModel,
        parameters: List[Parameter],
        params: torch.Tensor,
        momentum: torch.Tensor,
    ):
        pass


@register_class
class LeapfrogIntegrator(JSONSerializable, Integrator):
    def __init__(self, steps: int, step_size: float):
        self.steps = steps
        self.step_size = step_size

    def __call__(
        self,
        model: CallableModel,
        parameters: List[Parameter],
        momentum: torch.Tensor,
        inverse_mass_matrix: torch.Tensor,
    ):
        params = torch.cat([parameter.tensor.clone() for parameter in parameters], -1)
        momentum = momentum.clone()
        set_tensor(parameters, params.detach())
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

        set_tensor(parameters, params.detach())

        momentum -= self.step_size / 2 * dU
        return momentum

    @classmethod
    def from_json(cls, data, dic):
        step_size = data.get('step_size', 0.01)
        steps = data.get('steps', 10)
        return cls(steps, step_size)
