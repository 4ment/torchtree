from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal

from ...core.model import CallableModel
from ...core.parameter_utils import pack_tensor, save_parameters
from ...core.runnable import Runnable
from ...core.serializable import JSONSerializable
from ...core.utils import process_object, process_objects, register_class
from ...typing import ListParameter
from ..utils import extract_tensors_and_parameters
from .integrator import Integrator


@register_class
class HMC(JSONSerializable, Runnable):
    def __init__(
        self,
        parameters: ListParameter,
        joint: CallableModel,
        iterations: int,
        integrator: Integrator,
        **kwargs,
    ) -> None:
        self.parameters = parameters
        self.joint = joint
        self.iterations = iterations
        self.integrator = integrator
        self.dimension = sum(
            [parameter.tensor.shape[0] for parameter in self.parameters]
        )

        if 'mass_matrix' in kwargs:
            self.mass_matrix = kwargs['mass_matrix']
            self.inverse_mass_matrix = 1.0 / self.mass_matrix  # only works for diagonal
            self.sqrt_mass_matrix = self.mass_matrix.sqrt()
        else:
            self.mass_matrix = torch.ones(self.dimension)
            self.inverse_mass_matrix = torch.ones(self.dimension)
            self.sqrt_mass_matrix = torch.ones(self.dimension)

        self.warmup_adaptor = kwargs.get('adaptation', None)
        self.loggers = kwargs.get('loggers', ())
        self.checkpoint = kwargs.get('checkpoint', None)
        self.checkpoint_frequency = kwargs.get('checkpoint_frequency', 1000)
        self.every = kwargs.get('every', 100)
        self.find_step_size = kwargs.get('find_set_size', True)

    def sample_momentum(self, params):
        if self.sqrt_mass_matrix.dim() == 1:
            momentum = Normal(
                torch.zeros(params.size()),
                self.sqrt_mass_matrix,
            ).sample()
        else:
            momentum = MultivariateNormal(
                torch.zeros(params.size()),
                self.mass_matrix,
            ).sample()
        return momentum

    def hamiltonian(self, momentum):
        with torch.no_grad():
            potential_energy = -self.joint()
        # diagonal mass matrix
        if self.inverse_mass_matrix.dim() == 1:
            kinetic_energy = (
                torch.dot(momentum, self.inverse_mass_matrix * momentum) * 0.5
            )
        else:
            kinetic_energy = (
                torch.dot(momentum, self.inverse_mass_matrix @ momentum) * 0.5
            )
        hamiltonian = potential_energy + kinetic_energy
        return hamiltonian

    def run(self) -> None:
        accept = 0

        for logger in self.loggers:
            logger.initialize()

        if self.find_step_size:
            self.find_reasonable_step_size()

        if self.warmup_adaptor is not None:
            self.warmup_adaptor.initialize(self.integrator.step_size, self.mass_matrix)

        print('  iter             logP   hamiltonian   accept ratio   step size ')

        for epoch in range(1, self.iterations + 1):
            params = torch.cat(
                [parameter.tensor.clone() for parameter in self.parameters], -1
            )
            momentum = self.sample_momentum(params)
            hamiltonian = self.hamiltonian(momentum)

            momentum = self.integrator(
                self.joint, self.parameters, momentum, self.inverse_mass_matrix
            )
            proposed_hamiltonian = self.hamiltonian(momentum)

            alpha = -proposed_hamiltonian + hamiltonian

            rho = min(0.0, alpha)
            if rho > np.log(np.random.uniform()):
                accept += 1
            else:
                pack_tensor(self.parameters, params)

            if epoch % self.every == 0:
                print(
                    '  {:>4}  {:>15.3f}  {:>12.3f}  {:>13.3f}    {:.6f}'.format(
                        epoch,
                        self.joint(),
                        proposed_hamiltonian,
                        accept / epoch,
                        self.integrator.step_size,
                    )
                )

            for logger in self.loggers:
                logger.log(sample=epoch)

            if self.checkpoint is not None and epoch % self.checkpoint_frequency == 0:
                save_parameters(self.checkpoint, self.parameters)

            if self.warmup_adaptor is not None:
                self.warmup_adaptor.learn(params, torch.exp(alpha))
                self.integrator.step_size = self.warmup_adaptor.step_size
                self.mass_matrix = self.warmup_adaptor.mass_matrix
                self.inverse_mass_matrix = self.warmup_adaptor.inverse_mass_matrix
                self.sqrt_mass_matrix = self.warmup_adaptor.sqrt_mass_matrix

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    def find_reasonable_step_size(self):
        direction_threshold = math.log(0.8)
        params = torch.cat(
            [parameter.tensor.clone() for parameter in self.parameters], -1
        )
        r = self.sample_momentum(params)
        hamiltonian = self.hamiltonian(r)

        r = self.integrator(self.joint, self.parameters, r, self.inverse_mass_matrix)
        new_hamiltonian = self.hamiltonian(r)

        delta_hamiltonian = hamiltonian - new_hamiltonian
        direction = 1 if direction_threshold < delta_hamiltonian else -1

        while True:
            r = self.sample_momentum(params)
            hamiltonian = self.hamiltonian(r)

            r = self.integrator(
                self.joint, self.parameters, r, self.inverse_mass_matrix
            )
            new_hamiltonian = self.hamiltonian(r)

            delta_hamiltonian = hamiltonian - new_hamiltonian

            if (direction == 1 and delta_hamiltonian <= direction_threshold) or (
                direction == -1 and delta_hamiltonian >= direction_threshold
            ):
                break
            else:
                self.integrator.step_size = self.integrator.step_size * (
                    2.0**direction
                )

    @classmethod
    def from_json(cls, data: dict[str, any], dic: dict[str, any]) -> HMC:
        iterations = data['iterations']

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

        integrator = process_object(data['integrator'], dic)

        if 'adaptation' in data:
            optionals['adaptation'] = process_object(data['adaptation'], dic)

        if 'every' in data:
            optionals['every'] = data['every']

        if 'find_step_size' in data:
            optionals['find_step_size'] = data['find_step_size']

        return cls(parameters, joint, iterations, integrator, **optionals)
