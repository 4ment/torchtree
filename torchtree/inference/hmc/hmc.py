from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import Normal

from ...core.model import CallableModel
from ...core.parameter_utils import save_parameters
from ...core.runnable import Runnable
from ...core.serializable import JSONSerializable
from ...core.utils import JSONParseError, process_objects, register_class
from ...typing import ListParameter
from ..utils import extract_tensors_and_parameters
from .adaptation import StepSizeAdaptation


@register_class
class HMC(JSONSerializable, Runnable):
    def __init__(
        self,
        parameters: ListParameter,
        joint: CallableModel,
        iterations: int,
        steps: int,
        step_size: int,
        **kwargs,
    ) -> None:
        self.parameters = parameters
        self.joint = joint
        self.iterations = iterations
        self.steps = steps
        self.step_size = step_size
        self.step_size_adaptor = kwargs.get('step_size_adaptor', None)
        self.loggers = kwargs.get('loggers', ())
        self.checkpoint = kwargs.get('checkpoint', None)
        self.checkpoint_frequency = kwargs.get('checkpoint_frequency', 1000)
        self.every = kwargs.get('every', 100)

    def set_tensor(self, tensor: torch.Tensor) -> None:
        start = 0
        for parameter in self.parameters:
            parameter.tensor = tensor[
                ..., start : (start + parameter.shape[-1])
            ].requires_grad_()
            start += parameter.shape[-1]

    def sample_momentum(self, params):
        momentum = Normal(
            torch.zeros(params.size()),
            torch.ones(params.size()),
        ).sample()
        return momentum

    def leapfrog(
        self, params: torch.Tensor, momentum: torch.Tensor, steps: int, step_size: float
    ):
        momentum = momentum.clone()
        params = params.clone()
        self.set_tensor(params.detach())
        U = self.joint()
        U.backward()
        dU = -torch.cat([parameter.grad for parameter in self.parameters], -1)

        momentum = momentum - step_size / 2.0 * dU

        for _ in range(steps):
            params = params + step_size * momentum
            self.set_tensor(params.detach())
            U = self.joint()
            U.backward()
            dU = -torch.cat([parameter.grad for parameter in self.parameters], -1)
            momentum -= step_size * dU

        self.set_tensor(params.detach())

        momentum -= step_size / 2 * dU
        return momentum

    def hamiltonian(self, momentum):
        with torch.no_grad():
            potential_energy = -self.joint()
        kinetic_energy = torch.dot(momentum, momentum) * 0.5
        hamiltonian = potential_energy + kinetic_energy
        return hamiltonian

    def run(self) -> None:
        accept = 0

        for logger in self.loggers:
            logger.initialize()

        print('  iter             logP   hamiltonian   accept ratio   step size ')

        for epoch in range(1, self.iterations + 1):
            params = torch.cat(
                [parameter.tensor.clone() for parameter in self.parameters], -1
            )
            momentum = self.sample_momentum(params)
            hamiltonian = self.hamiltonian(momentum)

            momentum = self.leapfrog(params, momentum, self.steps, self.step_size)
            proposed_hamiltonian = self.hamiltonian(momentum)

            alpha = -proposed_hamiltonian + hamiltonian

            rho = min(0.0, alpha)
            if rho > np.log(np.random.uniform()):
                accept += 1
            else:
                self.set_tensor(params)

            if epoch % self.every == 0:
                print(
                    '  {:>4}  {:>15.3f}  {:>12.3f}  {:>13.3f}    {:.6f}'.format(
                        epoch,
                        self.joint(),
                        proposed_hamiltonian,
                        accept / epoch,
                        self.step_size,
                    )
                )

            for logger in self.loggers:
                logger.log()

            if self.checkpoint is not None and epoch % self.checkpoint_frequency == 0:
                save_parameters(self.checkpoint, self.parameters)

            if self.step_size_adaptor is not None:
                if epoch < 1000:
                    self.step_size = self.step_size_adaptor.learn_stepsize(
                        math.exp(alpha)
                    )
                elif epoch == 1000:
                    self.step_size = self.step_size_adaptor.complete_adaptation()

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    @classmethod
    def from_json(cls, data: dict[str, any], dic: dict[str, any]) -> HMC:
        iterations = data['iterations']
        steps = data['steps']
        step_size = data['step_size']

        optionals = {}
        # checkpointing is used by default and the default file name is checkpoint.json
        # it can be disabled if 'checkpoint': false is used
        # the name of the checkpoint file can be modified using
        # 'checkpoint': 'checkpointer.json'
        if 'checkpoint' in data:
            if isinstance(data['checkpoint'], bool) and data['checkpoint']:
                optionals['checkpoint'] = 'checkpoint.json'
            elif isinstance(data['checkpoint'], str):
                optionals['checkpoint'] = data['checkpoint']
        else:
            optionals['checkpoint'] = 'checkpoint.json'

        if 'checkpoint_frequency' in data:
            optionals['checkpoint_frequency'] = data['checkpoint_frequency']

        if 'loggers' in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals['loggers'] = loggers

        joint = process_objects(data['joint'], dic)

        _, parameters = extract_tensors_and_parameters(data['parameters'], dic)

        if 'step_size_adaptor' in data:
            if (
                isinstance(data['step_size_adaptor'], bool)
                and data['step_size_adaptor']
            ):
                optionals['step_size_adaptor'] = StepSizeAdaptation(
                    mu=math.log(10.0 * step_size)
                )
            elif isinstance(data['step_size_adaptor'], dict):
                optionals['step_size_adaptor'] = StepSizeAdaptation(
                    **data['step_size_adaptor']
                )
            else:
                raise JSONParseError(
                    f"'step_size_adaptor' element in HMC object ({data['id']}) "
                    "should be a boolean or a dictionary"
                )
        if 'every' in data:
            optionals['every'] = data['every']

        return cls(parameters, joint, iterations, steps, step_size, **optionals)
