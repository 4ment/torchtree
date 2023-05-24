# import abc
# import math

# import torch
# from torch import Tensor

# from torchtree.ops.dual_averaging import DualAveraging
# from torchtree.ops.welford import WelfordVariance
# from torchtree.typing import ListParameter

# from ...core.abstractparameter import AbstractParameter
# from ...core.serializable import JSONSerializable
# from ...core.utils import process_object, register_class
# from ..utils import extract_tensors_and_parameters
# from .integrator import LeapfrogIntegrator


# class Adaptor(JSONSerializable, abc.ABC):
#     @abc.abstractmethod
#     def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
#         ...

#     @abc.abstractmethod
#     def restart(self) -> None:
#         pass


# @register_class
# class AdaptiveStepSize(Adaptor):
#     def __init__(
#         self,
#         integrator: LeapfrogIntegrator,
#         target_acceptance_probability: float,
#         **kwargs
#     ):
#         self._integrator = integrator
#         self.target_acceptance_probability = target_acceptance_probability
#         self._delay = kwargs.get("delay", 0)
#         self._maximum = kwargs.get("maximum", float("inf"))
#         self._call_counter = 0
#         self._accepted = 0
#         self._acceptance_rate = kwargs.get("use_acceptance_rate ", False)

#     def restart(self) -> None:
#         pass

#     def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
#         self._call_counter += 1
#         self._accepted += accepted

#         if self._delay < self._call_counter < self._maximum and (
#             not self._acceptance_rate or self._call_counter >= 10
#         ):
#             prob = (
#                 self._accepted / self._call_counter
#                 if self._acceptance_rate
#                 else acceptance_prob
#             )
#             new_parameter = math.log(self._integrator.step_size) + (
#                 prob - self.target_acceptance_probability
#             ) / (2 + self._call_counter)
#             self._integrator.step_size = math.exp(new_parameter)

#     @classmethod
#     def from_json(cls, data, dic):
#         integrator = process_object(data['integrator'], dic)
#         target_acceptance_probability = data.get('target_acceptance_probability', 0.8)
#         options = {}
#         if 'delay' in data:
#             options['delay'] = data['delay']
#         if 'maximum' in data:
#             options['maximum'] = data['maximum']
#         options['use_acceptance_rate'] = data.get("use_acceptance_rate", False)
#         return cls(integrator, target_acceptance_probability, **options)


# @register_class
# class DualAveragingStepSize(Adaptor):
#     r"""Step size adaptation using dual averaging Nesterov.

#     Code adapted from: https://github.com/stan-dev/stan
#     """

#     def __init__(
#         self,
#         integrator: LeapfrogIntegrator,
#         mu=0.5,
#         delta=0.8,
#         gamma=0.05,
#         kappa=0.75,
#         t0=10,
#         **kwargs
#     ):
#         self.integrator = integrator
#         self._dual_avg = DualAveraging(mu=mu, gamma=gamma, kappa=kappa, t0=t0)
#         self._delta = delta
#         self._delay = kwargs.get("delay", 0)
#         self._maximum = kwargs.get("maximum", float("inf"))
#         self._call_counter = 0
#         self.restart()

#     def restart(self) -> None:
#         self._dual_avg.restart()

#     def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
#         self._call_counter += 1

#         if self._delay < self._call_counter < self._maximum:
#             self._dual_avg.step(self._delta - acceptance_prob)
#             self.integrator.step_size = math.exp(self._dual_avg.x)
#         elif self._call_counter >= self._maximum:
#             self.integrator.step_size = math.exp(self._dual_avg.x_bar)

#     @classmethod
#     def from_json(cls, data, dic):
#         integrator = process_object(data['integrator'], dic)
#         if "mu" in data:
#             mu = data["mu"]
#         else:
#             mu = math.log(10.0 * integrator.step_size)
#         target_acceptance_probability = data.get('target_acceptance_probability', 0.8)
#         gamma = data.get('gamma', 0.05)
#         kappa = data.get('kappa', 0.75)
#         t0 = data.get('t0', 10)
#         options = {}
#         if 'delay' in data:
#             options['delay'] = data['delay']
#         if 'maximum' in data:
#             options['maximum'] = data['maximum']
#         return cls(
#             integrator,
#             mu=mu,
#             delta=target_acceptance_probability,
#             gamma=gamma,
#             kappa=kappa,
#             t0=t0,
#             **options
#         )


# @register_class
# class MassMatrixAdaptor(Adaptor):
#     def __init__(
#         self,
#         parameters: ListParameter,
#         mass_matrix: AbstractParameter,
#         regularize=True,
#         **kwargs
#     ):
#         self._mass_matrix = mass_matrix
#         self._parameters = parameters
#         dim = mass_matrix.shape[0]
#         self._diagonal = mass_matrix.tensor.dim() == 1
#         self.variance_estimator = WelfordVariance(
#             torch.zeros([dim]), torch.zeros_like(mass_matrix.tensor)
#         )

#         self._regularize = regularize
#         self._delay = kwargs.get("delay", 0)
#         self._maximum = kwargs.get("maximum", float("inf"))
#         self._frequency = kwargs.get("update_frequency", 10)
#         self._restart_frequency = kwargs.get("restart_frequency", float("inf"))
#         self._call_counter = 0

#     def learn(self, acceptance_prob: Tensor, sample: int, accepted: bool) -> None:
#         self._call_counter += 1

#         if self._delay < self._call_counter < self._maximum:
#             if self._call_counter % self._restart_frequency == 0:
#                 self.variance_estimator.reset()
#                 return

#             x = torch.cat([parameter.tensor for parameter in self._parameters], -1)
#             self.variance_estimator.add_sample(x)
#             if self._call_counter % self._frequency == 0:
#                 inverse_mass_matrix = self.variance_estimator.variance()
#                 if self._regularize:
#                     n = self.variance_estimator.samples
#                     inverse_mass_matrix *= n / (n + 5.0)
#                     if self._diagonal:
#                         inverse_mass_matrix += 1e-3 * (5.0 / (n + 5.0))
#                     else:
#                         dim = inverse_mass_matrix.shape[0]
#                         inverse_mass_matrix[range(dim), range(dim)] += 1e-3 * (
#                             5.0 / (n + 5.0)
#                         )
#                 if self._diagonal:
#                     self._mass_matrix.tensor = 1.0 / inverse_mass_matrix
#                 else:
#                     self._mass_matrix.tensor = torch.inverse(inverse_mass_matrix)

#     def restart(self) -> None:
#         self.variance_estimator.reset()

#     @classmethod
#     def from_json(cls, data, dic):
#         _, parameters = extract_tensors_and_parameters(data['parameters'], dic)
#         mass_matrix = process_object(data['mass_matrix'], dic)
#         dimension = sum([parameter.tensor.shape[0] for parameter in parameters])
#         if mass_matrix.tensor.shape[0] != dimension:
#             mass_matrix.tensor = mass_matrix.tensor.reshape(dimension)
#         options = {}
#         if 'delay' in data:
#             options['delay'] = data['delay']
#         if 'maximum' in data:
#             options['maximum'] = data['maximum']
#         if 'update_frequency' in data:
#             options['update_frequency'] = data['update_frequency']
#         if 'restart_frequency' in data:
#             options['restart_frequency'] = data['restart_frequency']
#         return cls(parameters, mass_matrix, data.get('regularize', True), **options)
