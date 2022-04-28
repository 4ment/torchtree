from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.distributions import Normal

from torchtree.core.model import CallableModel
from torchtree.core.parameter_encoder import ParameterEncoder
from torchtree.core.runnable import Runnable
from torchtree.core.serializable import JSONSerializable
from torchtree.core.utils import process_objects, register_class
from torchtree.optim.optimizer import Optimizer
from torchtree.typing import ListParameter


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
        self.loggers = kwargs.get('loggers', ())
        self.checkpoint = kwargs.get('checkpoint', None)
        self.checkpoint_frequency = kwargs.get('checkpoint_frequency', 1000)

    def update_checkpoint(self) -> None:
        if not os.path.lexists(self.checkpoint):
            with open(self.checkpoint, 'w') as fp:
                json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
        else:
            with open(self.checkpoint + '.new', 'w') as fp:
                json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
            os.rename(self.checkpoint, self.checkpoint + '.old')
            os.rename(self.checkpoint + '.new', self.checkpoint)
            os.remove(self.checkpoint + '.old')

    def set_tensor(self, tensor: torch.Tensor) -> None:
        start = 0
        for parameter in self.parameters:
            parameter.tensor = tensor[
                ..., start : (start + parameter.shape[-1])
            ].requires_grad_()
            start += parameter.shape[-1]

    def leapfrog(self, params: torch.Tensor, steps: int, step_size: float):
        x0 = params
        p0 = Normal(
            torch.zeros(params.size()),
            torch.ones(params.size()),
        ).sample()
        self.set_tensor(params.detach())
        U = self.joint()  # func(params)
        U.backward()
        dU = -torch.cat([parameter.grad for parameter in self.parameters], -1)

        # Half step for momentum
        p_step = p0 - step_size / 2.0 * dU

        # Full step for position
        x_step = x0 + step_size * p_step

        for _ in range(steps):
            self.set_tensor(x_step.detach())
            U = self.joint()
            U.backward()
            dU = -torch.cat([parameter.grad for parameter in self.parameters], -1)
            # Update momentum
            p_step -= step_size * dU

            # Update position
            x_step += step_size * p_step

        self.set_tensor(x_step.detach())

        # Half for momentum
        p_step -= step_size / 2 * dU
        return torch.dot(p0, p0) / 2 - torch.dot(p_step, p_step) / 2

    def run(self) -> None:
        with torch.no_grad():
            logP = self.joint()
        accept = 0

        for logger in self.loggers:
            logger.initialize()

        for epoch in range(1, self.iterations + 1):
            beta_prev = torch.cat(
                [parameter.tensor for parameter in self.parameters], -1
            )
            kinetic_diff = self.leapfrog(beta_prev, self.steps, self.step_size)
            with torch.no_grad():
                proposed_logP = self.joint()

            alpha = proposed_logP - logP + kinetic_diff

            print(logP, proposed_logP, kinetic_diff, alpha, accept / epoch)

            if alpha >= 0 or alpha > np.log(np.random.uniform()):
                logP = proposed_logP
                accept += 1
            else:
                self.set_tensor(beta_prev)

            for logger in self.loggers:
                logger.log()

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

        if 'loggers' in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals['loggers'] = loggers

        joint = process_objects(data['joint'], dic)

        _, parameters = Optimizer.parse_params(data['parameters'], dic)

        return cls(parameters, joint, iterations, steps, step_size, **optionals)
