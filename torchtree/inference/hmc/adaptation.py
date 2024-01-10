from __future__ import annotations

import abc
import copy
import math
from collections import deque
from typing import Any

import torch
from torch import Tensor

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.identifiable import Identifiable
from torchtree.core.utils import process_object, register_class
from torchtree.inference.hmc.integrator import LeapfrogIntegrator
from torchtree.inference.utils import extract_tensors_and_parameters
from torchtree.ops.dual_averaging import DualAveraging
from torchtree.ops.welford import WelfordVariance
from torchtree.typing import ID, ListParameter


class Adaptor(Identifiable, abc.ABC):
    def __init__(self, id_):
        Identifiable.__init__(self, id_)

    @abc.abstractmethod
    def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        ...

    @abc.abstractmethod
    def restart(self) -> None:
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
class AdaptiveStepSize(Adaptor):
    def __init__(
        self,
        id_: ID,
        integrator: LeapfrogIntegrator,
        target_acceptance_probability: float,
        **kwargs,
    ):
        Adaptor.__init__(self, id_)
        self._integrator = integrator
        self.target_acceptance_probability = target_acceptance_probability
        self._start = kwargs.get("start", 1)
        self._end = kwargs.get("end", float("inf"))
        self._call_counter = 0
        self._accepted = 0
        self._acceptance_rate = kwargs.get("use_acceptance_rate", False)

    def restart(self) -> None:
        pass

    def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        self._call_counter += 1
        self._accepted += accepted

        if self._start <= self._call_counter <= self._end and (
            not self._acceptance_rate or self._call_counter >= 10
        ):
            prob = (
                self._accepted / self._call_counter
                if self._acceptance_rate
                else acceptance_prob
            )

            new_parameter = math.log(self._integrator.step_size) + (
                prob - self.target_acceptance_probability
            ) / (2 + self._call_counter)
            self._integrator.step_size = math.exp(new_parameter)

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {
            "call_counter": self._call_counter,
            "accepted": self._accepted,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._call_counter = state_dict["call_counter"]
        self._accepted = state_dict["accepted"]

    @classmethod
    def from_json(cls, data, dic):
        integrator = process_object(data["integrator"], dic)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.8)
        options = {}
        if "start" in data:
            options["start"] = data["start"]
        if "end" in data:
            options["end"] = data["end"]
        options["use_acceptance_rate"] = data.get("use_acceptance_rate", False)
        return cls(data["id"], integrator, target_acceptance_probability, **options)


@register_class
class DualAveragingStepSize(Adaptor):
    r"""Step size adaptation using dual averaging Nesterov.

    Code adapted from: https://github.com/stan-dev/stan
    """

    def __init__(
        self,
        id_: ID,
        integrator: LeapfrogIntegrator,
        mu=0.5,
        delta=0.8,
        gamma=0.05,
        kappa=0.75,
        t0=10,
        **kwargs,
    ):
        Adaptor.__init__(self, id_)
        self.integrator = integrator
        self._dual_avg = DualAveraging(mu=mu, gamma=gamma, kappa=kappa, t0=t0)
        self._delta = delta
        self._start = kwargs.get("start", 0)
        self._end = kwargs.get("end", float("inf"))
        self._call_counter = 0
        self.restart()

    def restart(self) -> None:
        self._dual_avg.restart()

    def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        self._call_counter += 1

        if self._start <= self._call_counter <= self._end:
            self._dual_avg.step(self._delta - acceptance_prob)
            self.integrator.step_size = math.exp(self._dual_avg.x)
        elif self._call_counter >= self._end:
            self.integrator.step_size = math.exp(self._dual_avg.x_bar)

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {
            "call_counter": self._call_counter,
        }
        state_dict_dual = {
            "counter": self._dual_avg._counter,
            "x": self._dual_avg.x,
            "x_bar": self._dual_avg.x_bar,
            "s_bar": self._dual_avg.s_bar,
        }
        state_dict.update(state_dict_dual)
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._call_counter = state_dict["call_counter"]
        self._accepted = state_dict["accepted"]
        self._dual_avg.x = state_dict["x"]
        self._dual_avg.x_bar = state_dict["x_bar"]
        self._dual_avg.s_bar = state_dict["s_bar"]

    @classmethod
    def from_json(cls, data, dic):
        integrator = process_object(data["integrator"], dic)
        mu = data.get("mu", math.log(10.0 * integrator.step_size))
        target_acceptance_probability = data.get("target_acceptance_probability", 0.8)
        gamma = data.get("gamma", 0.05)
        kappa = data.get("kappa", 0.75)
        t0 = data.get("t0", 10)
        options = {}
        if "start" in data:
            options["start"] = data["start"]
        if "end" in data:
            options["end"] = data["end"]
        return cls(
            data["id"],
            integrator,
            mu=mu,
            delta=target_acceptance_probability,
            gamma=gamma,
            kappa=kappa,
            t0=t0,
            **options,
        )


@register_class
class MassMatrixAdaptor(Adaptor):
    def __init__(
        self,
        id_: ID,
        parameters: ListParameter,
        mass_matrix: AbstractParameter,
        regularize=True,
        **kwargs,
    ):
        Adaptor.__init__(self, id_)
        self._mass_matrix = mass_matrix
        self._parameters = parameters
        dim = mass_matrix.shape[0]
        self._diagonal = mass_matrix.tensor.dim() == 1
        self.variance_estimator = WelfordVariance(
            torch.zeros([dim]), torch.zeros_like(mass_matrix.tensor)
        )

        self._regularize = regularize
        self._start = kwargs.get("start", 0)
        self._end = kwargs.get("end", float("inf"))
        self._frequency = kwargs.get("update_frequency", 10)
        self._restart_frequency = kwargs.get("restart_frequency", float("inf"))
        self._variance_window = kwargs.get("variance_window", 0)
        self._swap_every = kwargs.get("swap_every", 0)
        self.variance_estimator2 = None
        if self._swap_every != 0:
            self.variance_estimator2 = WelfordVariance(
                torch.zeros([dim]), torch.zeros_like(mass_matrix.tensor)
            )
        self._call_counter = 0
        self._values = deque()

    @property
    def mass_matrix(self):
        return self._mass_matrix

    def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
        self._call_counter += 1

        if self._start <= self._call_counter <= self._end:
            if self._call_counter % self._restart_frequency == 0:
                self.variance_estimator.reset()
                return

            x = torch.cat(
                [parameter.tensor.detach().clone() for parameter in self._parameters],
                -1,
            )

            self.variance_estimator.add_sample(x)

            if self._variance_window != 0:
                self._values.append(x)
                if len(self._values) > 100:
                    removed = self._values.popleft()
                    self.variance_estimator.remove_sample(removed)
            elif self._swap_every != 0:
                self.variance_estimator2.add_sample(x)

            if (
                self._call_counter % self._frequency == 0
                and self.variance_estimator.samples > 4
            ):
                inverse_mass_matrix = self.variance_estimator.variance()
                if self._regularize:
                    n = self.variance_estimator.samples
                    inverse_mass_matrix *= n / (n + 5.0)
                    if self._diagonal:
                        inverse_mass_matrix += 1e-3 * (5.0 / (n + 5.0))
                    else:
                        dim = inverse_mass_matrix.shape[0]
                        inverse_mass_matrix[range(dim), range(dim)] += 1e-3 * (
                            5.0 / (n + 5.0)
                        )
                if self._diagonal:
                    self._mass_matrix.tensor = 1.0 / inverse_mass_matrix
                else:
                    self._mass_matrix.tensor = torch.inverse(inverse_mass_matrix)

            if self._swap_every != 0 and self._call_counter % self._swap_every == 0:
                self.variance_estimator = copy.deepcopy(self.variance_estimator2)
                self.variance_estimator2.reset()

    def restart(self) -> None:
        self.variance_estimator.reset()

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {
            "call_counter": self._call_counter,
        }
        state_dict_estimator = {
            "mean": self.variance_estimator._mean.tolist(),
            "variance": self.variance_estimator._variance.tolist(),
            "samples": self.variance_estimator.samples,
        }
        state_dict.update(state_dict_estimator)
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._call_counter = state_dict["call_counter"]
        self.variance_estimator.samples = state_dict["samples"]
        info = {
            "dtype": self.variance_estimator._mean.dtype,
            "device": self.variance_estimator._mean.device,
        }
        self.variance_estimator._mean = torch.tensor(state_dict["mean"], **info)
        self.variance_estimator._variance = torch.tensor(state_dict["variance"], **info)

    @classmethod
    def from_json(cls, data, dic):
        _, parameters = extract_tensors_and_parameters(data["parameters"], dic)
        mass_matrix = process_object(data["mass_matrix"], dic)

        regularize = data.get("regularize", True)
        options = {}
        if "start" in data:
            options["start"] = data["start"]
        if "end" in data:
            options["end"] = data["end"]
        if "update_frequency" in data:
            options["update_frequency"] = data["update_frequency"]
        if "restart_frequency" in data:
            options["restart_frequency"] = data["restart_frequency"]
        if "variance_window" in data:
            options["variance_window"] = data["variance_window"]
        elif "swap_every" in data:
            options["swap_every"] = data["swap_every"]

        return cls(data["id"], parameters, mass_matrix, regularize, **options)


def find_reasonable_step_size(
    integrator, parameters, hamiltonian, mass_matrix, inverse_mass_matrix
):
    direction_threshold = math.log(0.8)
    r = hamiltonian.sample_momentum(mass_matrix)
    ham = hamiltonian(momentum=r, inverse_mass_matrix=inverse_mass_matrix)

    r = integrator(hamiltonian.joint, parameters, r, inverse_mass_matrix)

    new_ham = hamiltonian(momentum=r, inverse_mass_matrix=inverse_mass_matrix)

    delta_hamiltonian = ham - new_ham
    direction = 1 if direction_threshold < delta_hamiltonian else -1

    while True:
        r = hamiltonian.sample_momentum(mass_matrix)
        ham = hamiltonian(momentum=r, inverse_mass_matrix=inverse_mass_matrix)

        r = integrator(hamiltonian.joint, parameters, r, inverse_mass_matrix)

        new_ham = hamiltonian(momentum=r, inverse_mass_matrix=inverse_mass_matrix)

        delta_hamiltonian = ham - new_ham

        if (direction == 1 and delta_hamiltonian <= direction_threshold) or (
            direction == -1 and delta_hamiltonian >= direction_threshold
        ):
            break
        else:
            integrator.step_size = integrator.step_size * (2.0**direction)


class WarmupAdaptation(Adaptor):
    @property
    @abc.abstractmethod
    def step_size(self):
        ...

    @property
    @abc.abstractmethod
    def mass_matrix(self):
        ...

    @property
    @abc.abstractmethod
    def inverse_mass_matrix(self):
        ...

    @property
    @abc.abstractmethod
    def sqrt_mass_matrix(self):
        ...
