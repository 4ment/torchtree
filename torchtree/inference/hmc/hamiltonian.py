from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from ...core.model import CallableModel
from ...core.utils import process_object, register_class
from ...typing import ID


@register_class
class Hamiltonian(CallableModel):
    def __init__(self, id_: ID, joint: CallableModel):
        super().__init__(id_)
        self.joint = joint

    def _call(self, *args, **kwargs) -> Tensor:
        momentum: Tensor = kwargs["momentum"]
        if "inverse_mass_matrix" in kwargs:
            inverse_mass_matrix: Tensor = kwargs["inverse_mass_matrix"]
        elif "mass_matrix" in kwargs:
            mass_matrix: Tensor = kwargs["mass_matrix"]
            if mass_matrix.dim() == 1:
                inverse_mass_matrix = 1.0 / mass_matrix
            else:
                inverse_mass_matrix = torch.inverse(mass_matrix)
        kinetic_energy = self.kinetic_energy(momentum, inverse_mass_matrix)
        potential_energy = self.potential_energy()
        hamiltonian = potential_energy + kinetic_energy
        return hamiltonian

    def sample_momentum(self, mass_matrix: Tensor) -> None:
        if mass_matrix.dim() == 1:
            momentum = Normal(
                torch.zeros_like(mass_matrix),
                mass_matrix.sqrt(),
            ).sample()
        else:
            momentum = MultivariateNormal(
                torch.zeros(
                    mass_matrix.shape[0],
                    dtype=mass_matrix.dtype,
                    device=mass_matrix.device,
                ),
                covariance_matrix=mass_matrix,
            ).sample()
        return momentum

    def potential_energy(self) -> Tensor:
        with torch.no_grad():
            potential_energy = -self.joint()
        return potential_energy

    def kinetic_energy(self, momentum: Tensor, inverse_mass_matrix: Tensor) -> Tensor:
        # diagonal mass matrix
        if inverse_mass_matrix.dim() == 1:
            kinetic_energy = torch.dot(momentum, inverse_mass_matrix * momentum) * 0.5
        else:
            kinetic_energy = torch.dot(momentum, inverse_mass_matrix @ momentum) * 0.5
        return kinetic_energy

    def handle_parameter_changed(self, variable, index, event):
        pass

    def _sample_shape(self) -> torch.Size:
        return self.joint.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> Hamiltonian:
        joint = process_object(data["joint"], dic)
        return cls(data['id'], joint)
