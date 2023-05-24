import math


class DualAveraging:
    r"""Dual averaging Nesterov.

    Code adapted from: https://github.com/stan-dev/stan
    """

    def __init__(self, mu=0.5, gamma=0.05, kappa=0.75, t0=10):
        self._mu = mu
        self._gamma = gamma
        self._kappa = kappa
        self._t0 = t0
        self.x = None
        self.restart()

    def restart(self) -> None:
        self._counter = 0
        self.s_bar = 0
        self.x_bar = 0

    def step(self, statistic) -> None:
        self._counter += 1

        statistic = 1 if statistic > 1 else statistic

        # Nesterov Dual-Averaging of log(epsilon)
        eta = 1.0 / (self._counter + self._t0)

        self.s_bar = (1.0 - eta) * self.s_bar + eta * statistic

        self.x = self._mu - self.s_bar * math.sqrt(self._counter) / self._gamma
        x_eta = math.pow(self._counter, -self._kappa)

        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * self.x
