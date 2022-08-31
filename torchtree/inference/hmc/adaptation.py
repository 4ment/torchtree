import abc
import math

import torch
from torch.distributions import Normal


class StepSizeAdaptation:
    def __init__(self, mu=0.5, delta=0.8, gamma=0.05, kappa=0.75, t0=10):
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t0 = t0
        self.restart()

    def restart(self) -> None:
        self.counter = 0
        self.s_bar = 0
        self.x_bar = 0

    def learn_stepsize(self, adapt_stat):
        self.counter += 1

        adapt_stat = 1 if adapt_stat > 1 else adapt_stat

        # Nesterov Dual-Averaging of log(epsilon)
        eta = 1.0 / (self.counter + self.t0)

        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.delta - adapt_stat)

        x = self.mu - self.s_bar * math.sqrt(self.counter) / self.gamma
        x_eta = math.pow(self.counter, -self.kappa)

        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * x

        return math.exp(x)

    def complete_adaptation(self):
        return math.exp(self.x_bar)


class WelfordVariance:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor, samples=0) -> None:
        self.mean = mean
        self.variance = variance
        self.samples = samples

    def add_sample(self, x) -> None:
        self.samples += 1
        diff = x - self.mean
        self.mean += diff / self.samples
        self.variance += (x - self.mean) * diff


class MassMatrixAdaptor(abc.ABC):
    @property
    @abc.abstractmethod
    def inverse_mass_matrix(self) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def mass_matrix(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def update(self, x) -> None:
        ...

    @abc.abstractmethod
    def kinetic_energy(self, momentum) -> torch.Tensor:
        ...


class IdentityMassMatrix(MassMatrixAdaptor):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def mass_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim)

    def inverse_mass_matrix(self) -> torch.Tensor:
        return torch.eye(self.dim)

    def update(self, x):
        pass

    def sample(self) -> torch.Tensor:
        momentum = Normal(
            torch.zeros(self.dim),
            torch.ones(self.dim),
        ).sample()
        return momentum

    def kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        return torch.dot(momentum, momentum) * 0.5


class DiagonalMassMatrixAdaptor(MassMatrixAdaptor):
    def __init__(self, dim: int):
        self.variance_estimator = WelfordVariance(
            torch.zeros([dim]), torch.zeros([dim])
        )

    def mass_matrix(self):
        return 1.0 / self.variance_estimator.variance

    def inverse_mass_matrix(self):
        return self.variance_estimator.variance

    def update(self, x):
        self.variance_estimator.add_sample(x)

    def sample(self):
        momentum = Normal(
            torch.zeros(self.dim),
            self.mass_matrix.sqrt(),
        ).sample()
        return momentum

    def kinetic_energy(self, momentum):
        return torch.dot(momentum, self.inverse_mass_matrix * momentum) * 0.5
