import abc
import math

import torch

from ...core.serializable import JSONSerializable
from ...core.utils import process_object, register_class
from ..utils import extract_tensors_and_parameters


class Adaptation(JSONSerializable, abc.ABC):
    @abc.abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def restart(self) -> None:
        ...

    @abc.abstractmethod
    def learn(self) -> None:
        ...


@register_class
class StepSizeAdaptation(Adaptation):
    r"""Step size adaptation using dual averaging Nesterov.

    Code adapted from: https://github.com/stan-dev/stan
    """

    def __init__(self, mu=0.5, delta=0.8, gamma=0.05, kappa=0.75, t0=10):
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t0 = t0
        self._step_size = 1.0
        self.restart()

    def initialize(self, **kwargs):
        self.mu = kwargs.get('mu', self.mu)
        self._step_size = kwargs.get('step_size', self._step_size)

    @property
    def step_size(self) -> float:
        return self._step_size

    def restart(self) -> None:
        self.counter = 0
        self.s_bar = 0
        self.x_bar = 0

    def learn(self, adapt_stat):
        self.counter += 1

        adapt_stat = 1 if adapt_stat > 1 else adapt_stat

        # Nesterov Dual-Averaging of log(epsilon)
        eta = 1.0 / (self.counter + self.t0)

        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.delta - adapt_stat)

        x = self.mu - self.s_bar * math.sqrt(self.counter) / self.gamma
        x_eta = math.pow(self.counter, -self.kappa)

        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * x
        self._step_size = math.exp(x)

    def complete_adaptation(self):
        self._step_size = math.exp(self.x_bar)

    @classmethod
    def from_json(cls, data, dic):
        mu = data.get('mu', 0.5)
        delta = data.get('delta', 0.8)
        gamma = data.get('gamma', 0.05)
        kappa = data.get('kappa', 0.75)
        t0 = data.get('t0', 10)

        return cls(mu=mu, delta=delta, gamma=gamma, kappa=kappa, t0=t0)


class WelfordVariance:
    r"""Welford's online method for estimating variance."""

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor, samples=0) -> None:
        self._mean = mean
        self._variance = variance
        self.samples = samples

    def add_sample(self, x) -> None:
        self.samples += 1
        diff = x - self._mean
        self._mean += diff / self.samples
        self._variance += (x - self._mean) * diff

    def variance(self):
        return self._variance / (self.samples - 1)

    def mean(self):
        return self._mean

    def reset(self) -> None:
        self._mean = torch.zeros_like(self._mean)
        self._variance = torch.zeros_like(self._variance)
        self.samples = 0


class MassMatrixAdaptor(Adaptation):
    @property
    @abc.abstractmethod
    def inverse_mass_matrix(self) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def mass_matrix(self) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def sqrt_mass_matrix(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def update(self, x) -> None:
        ...


@register_class
class IdentityMassMatrix(MassMatrixAdaptor):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def mass_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim)

    def inverse_mass_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim)

    def sqrt_mass_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim)

    def update(self, x) -> None:
        pass

    def learn(self) -> None:
        pass

    def restart(self) -> None:
        pass

    @classmethod
    def from_json(cls, data, dic):
        _, parameters = extract_tensors_and_parameters(data['parameters'], dic)
        dimension = sum([parameter.tensor.shape[0] for parameter in parameters])
        return cls(dimension)


@register_class
class DiagonalMassMatrixAdaptor(MassMatrixAdaptor):
    def __init__(self, dim: int, regularize=True):
        self.variance_estimator = WelfordVariance(
            torch.zeros([dim]), torch.zeros([dim])
        )
        self._mass_matrix = torch.ones([dim])
        self._inverse_mass_matrix = torch.ones([dim])
        self._sqrt_mass_matrix = torch.ones([dim])
        self.regularize = regularize

    def initialize():
        pass

    @property
    def mass_matrix(self):
        return self._mass_matrix

    @property
    def inverse_mass_matrix(self):
        return self._inverse_mass_matrix

    @property
    def sqrt_mass_matrix(self):
        return self._sqrt_mass_matrix

    def learn(self):
        self._inverse_mass_matrix = self.variance_estimator.variance()
        if self.regularize:
            n = self.variance_estimator.samples
            self._inverse_mass_matrix = (
                n / (n + 5.0)
            ) * self._inverse_mass_matrix + 1e-3 * (5.0 / (n + 5.0))
        self._mass_matrix = 1.0 / self._inverse_mass_matrix
        self._sqrt_mass_matrix = self._mass_matrix.sqrt()

    def update(self, x):
        self.variance_estimator.add_sample(x)

    def restart(self) -> None:
        self.variance_estimator.reset()

    @classmethod
    def from_json(cls, data, dic):
        _, parameters = extract_tensors_and_parameters(data['parameters'], dic)
        dimension = sum([parameter.tensor.shape[0] for parameter in parameters])
        return cls(dimension, data.get('regularize', True))


class WarmupAdaptation(Adaptation):
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


@register_class
class StanWindowedAdaptation(WarmupAdaptation):
    r"""Adapts step size and mass matrix during a warmup period.

    Code adapted from Stan. See online manual for further details
    https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html

    :param step_size_adaptor: step size adaptor
    :param mass_matrix_adaptor: mass matrix adaptor
    :param int num_warmup: number of iteration of warmup period
    :param int init_buffer: width of initial fast adaptation interval
    :param int term_buffer: width of final fast adaptation interval
    :param int base window: initial width of slow adaptation interval
    """

    def __init__(
        self,
        step_size_adaptor: StepSizeAdaptation,
        mass_matrix_adaptor: MassMatrixAdaptor,
        num_warmup: int,
        init_buffer: int,
        term_buffer: int,
        base_window: int,
    ):
        self.num_warmup = 0
        self.adapt_init_buffer = 0
        self.adapt_term_buffer = 0
        self.adapt_base_window = 0
        self.step_size_adaptor = step_size_adaptor
        self.mass_matrix_adaptor = mass_matrix_adaptor
        self._step_size = None

        self.configure_window_parameters(
            num_warmup, init_buffer, term_buffer, base_window
        )

    def restart(self):
        self.adapt_window_counter = 0
        self.adapt_window_size = self.adapt_base_window
        self.adapt_next_window = self.adapt_init_buffer + self.adapt_window_size - 1

    def initialize(self, step_size, mass_matrix):
        if self.step_size_adaptor is not None:
            self.step_size_adaptor.initialize(
                mu=math.log(10.0 * step_size), step_size=step_size
            )
        self._step_size = step_size

    def configure_window_parameters(
        self, num_warmup, init_buffer, term_buffer, base_window
    ):
        if num_warmup < 20:
            print("WARNING: No estimation is")
            print("         performed for num_warmup < 20")
            exit(1)

        self.num_warmup = num_warmup
        if init_buffer + base_window + term_buffer > num_warmup:
            print("WARNING: There aren't enough warmup iterations to fit the")
            print("         three stages of adaptation as currently configured.")

            self.adapt_init_buffer = 0.15 * num_warmup
            self.adapt_term_buffer = 0.10 * num_warmup
            self.adapt_base_window = num_warmup - (
                self.adapt_init_buffer + self.adapt_term_buffer
            )

            print("         Reducing each adaptation stage to 15%/75%/10% of")
            print("         the given number of warmup iterations:")

            print(f"           init_buffer = {self.adapt_init_buffer}")
            print(f"           adapt_window = {self.adapt_base_window}")
            print(f"           term_buffer = {self.adapt_term_buffer}")
        else:
            self.adapt_init_buffer = init_buffer
            self.adapt_term_buffer = term_buffer
            self.adapt_base_window = base_window
        self.restart()

    def adaptation_window(self):
        return (
            (self.adapt_window_counter >= self.adapt_init_buffer)
            and (self.adapt_window_counter < self.num_warmup - self.adapt_term_buffer)
            and (self.adapt_window_counter != self.num_warmup)
        )

    def end_adaptation_window(self):
        return (
            self.adapt_window_counter == self.adapt_next_window
            and self.adapt_window_counter != self.num_warmup
        )

    def compute_next_window(self):
        if self.adapt_next_window == self.num_warmup - self.adapt_term_buffer - 1:
            return

        self.adapt_window_size *= 2
        self.adapt_next_window = self.adapt_window_counter + self.adapt_window_size

        if self.adapt_next_window == self.num_warmup - self.adapt_term_buffer - 1:
            return

        # Boundary of the following window, not the window just computed
        next_window_boundary = self.adapt_next_window + 2 * self.adapt_window_size

        # If the following window overtakes the full adaptation window,
        # then stretch the current window to the end of the full window
        if next_window_boundary >= self.num_warmup - self.adapt_term_buffer:
            self.adapt_next_window = self.num_warmup - self.adapt_term_buffer - 1

    def learn(self, z, adapt_stat):
        if self.adapt_window_counter >= self.num_warmup:
            if self.adapt_window_counter == self.num_warmup:
                self.step_size_adaptor.complete_adaptation()
            return

        if self.step_size_adaptor:
            self.step_size_adaptor.learn(adapt_stat)

        if self.adaptation_window():
            self.mass_matrix_adaptor.update(z)

        if self.end_adaptation_window():
            self.mass_matrix_adaptor.learn()
            self.step_size_adaptor.restart()
            self.mass_matrix_adaptor.restart()
            self.compute_next_window()

        self.adapt_window_counter += 1

    @property
    def step_size(self):
        if self.step_size_adaptor is None:
            return self._step_size
        else:
            return self.step_size_adaptor.step_size

    @property
    def mass_matrix(self):
        return self.mass_matrix_adaptor.mass_matrix

    @property
    def inverse_mass_matrix(self):
        return self.mass_matrix_adaptor.inverse_mass_matrix

    @property
    def sqrt_mass_matrix(self):
        return self.mass_matrix_adaptor.sqrt_mass_matrix

    @classmethod
    def from_json(cls, data, dic):
        warmup = data['warmup']
        initial = data['initial_window']
        final = data['final_window']
        base = data['base_window']
        step_size_adaptor = process_object(data['step_size_adaptor'], dic)
        mass_matrix_adaptor = process_object(data['mass_matrix_adaptor'], dic)
        return cls(step_size_adaptor, mass_matrix_adaptor, warmup, initial, final, base)
