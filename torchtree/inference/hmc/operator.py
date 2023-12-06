from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.model import CallableModel
from torchtree.core.parameter import Parameter
from torchtree.core.parametric import ParameterListener
from torchtree.core.utils import process_object, process_objects, register_class
from torchtree.inference.hmc.adaptation import Adaptor, find_reasonable_step_size
from torchtree.inference.hmc.hamiltonian import Hamiltonian
from torchtree.inference.hmc.integrator import Integrator
from torchtree.inference.mcmc.operator import MCMCOperator
from torchtree.typing import ID, ListParameter


@register_class
class HMCOperator(MCMCOperator, ParameterListener):
    def __init__(
        self,
        id_: ID,
        joint: CallableModel,
        parameters: ListParameter,
        integrator: Integrator,
        mass_matrix: AbstractParameter,
        weight: float = 1.0,
        target_acceptance_probability: float = 0.8,
        adaptors: list[Adaptor] = [],
        **kwargs,
    ):
        MCMCOperator.__init__(
            self,
            id_,
            parameters,
            weight,
            target_acceptance_probability,
            **kwargs,
        )
        self._integrator = integrator

        self._mass_matrix = mass_matrix
        self.update_mass_matrices()
        mass_matrix.add_parameter_listener(self)

        self._adaptors = adaptors

        self._hamiltonian = Hamiltonian(None, joint)

        if kwargs.get("find_reasonable_step_size", False):
            step_size = self._integrator.step_size
            find_reasonable_step_size(
                integrator,
                parameters,
                self._hamiltonian,
                self.mass_matrix,
                self.inverse_mass_matrix,
            )
            print(f"Step size: {self.id} = {self._integrator.step_size} ({step_size})")

        self._divergence_threshold = kwargs.get("divergence_threshold", 1000)

    def update_mass_matrices(self) -> None:
        if self.mass_matrix.dim() == 1:
            self.inverse_mass_matrix = 1.0 / self.mass_matrix
        else:
            self.inverse_mass_matrix = torch.inverse(self.mass_matrix)

    def handle_parameter_changed(
        self, variable: AbstractParameter, index, event
    ) -> None:
        self.update_mass_matrices()

    @property
    def mass_matrix(self) -> Tensor:
        return self._mass_matrix.tensor

    @property
    def tuning_parameter(self) -> float:
        return self._integrator.step_size

    @MCMCOperator.adaptable_parameter.getter
    def adaptable_parameter(self) -> Tensor:
        return math.log(self._integrator.step_size)

    def set_adaptable_parameter(self, value) -> None:
        self._integrator.step_size = math.exp(value)

    def _step(self) -> Tensor:
        for i in range(10):
            momentum = self._hamiltonian.sample_momentum(self.mass_matrix)
            ok = True
            try:
                kinetic_energy0 = self._hamiltonian.kinetic_energy(
                    momentum, self.inverse_mass_matrix
                )
                ham0 = self._hamiltonian.potential_energy() + kinetic_energy0

                momentum = self._integrator(
                    self._hamiltonian.joint,
                    self.parameters,
                    momentum,
                    self.inverse_mass_matrix,
                )

                kinetic_energy = self._hamiltonian.kinetic_energy(
                    momentum, self.inverse_mass_matrix
                )
                ham = self._hamiltonian.potential_energy() + kinetic_energy
            except ValueError:
                for parameter, saved_tensor in zip(self.parameters, self.saved_tensors):
                    parameter.tensor = saved_tensor
                ok = False
            if ok:
                break

        if not ok:
            return torch.tensor(float("inf"))

        if ham - ham0 > self._divergence_threshold:
            print(f"Hamiltonian divergence - {self.id}: {ham0} -> {ham} = {ham-ham0}")

        for parameter in self.parameters:
            parameter.requires_grad = False

        return kinetic_energy0 - kinetic_energy

    def tune(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        if len(self._adaptors) == 0:
            super().tune(acceptance_prob, sample, accepted)
        else:
            for adaptor in self._adaptors:
                adaptor.learn(acceptance_prob, sample, accepted)

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {
            "mass_matrix": self._mass_matrix,
        }
        if hasattr(self._integrator, "state_dict"):
            state_dict["integrator"] = self._integrator.state_dict()
        if len(self._adaptors) > 0:
            state_dict["adaptors"] = [
                adaptor.state_dict() for adaptor in self._adaptors
            ]
        return state_dict

    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        m = Parameter.from_json(state_dict["mass_matrix"], {})
        self._mass_matrix.tensor = m.tensor
        if hasattr(self._integrator, "load_state_dict"):
            self._integrator.load_state_dict(state_dict["integrator"])
        for adaptor in self._adaptors:
            for adaptor_state in state_dict["adaptors"]:
                if adaptor_state["id"] == adaptor.id:
                    adaptor.load_state_dict(adaptor_state)
                    break

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        joint = process_objects(data["joint"], dic)
        parameters = process_objects(data["parameters"], dic, force_list=True)
        integrator = process_object(data["integrator"], dic)
        mass_matrix = process_object(data["mass_matrix"], dic)
        adaptors = process_objects(data, dic, force_list=True, key="adaptors")
        weight = data.get("weight", 1.0)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.8)
        optionals = {}
        optionals["find_reasonable_step_size"] = data.get(
            "find_reasonable_step_size", False
        )
        optionals["divergence_threshold"] = data.get("divergence_threshold", 1000)
        if optionals["divergence_threshold"] in ("inf", "infinity"):
            optionals["divergence_threshold"] = float("inf")

        optionals["disable_adaptation"] = data.get("disable_adaptation", False)

        return cls(
            id_,
            joint,
            parameters,
            integrator,
            mass_matrix,
            weight,
            target_acceptance_probability,
            adaptors,
            **optionals,
        )
