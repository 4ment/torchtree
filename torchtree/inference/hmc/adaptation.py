import math


class StepSizeAdaptation:
    def __init__(self, mu=0.5, delta=0.8, gamma=0.05, kappa=0.75, t0=10):
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t0 = t0
        self.restart()

    def restart(self):
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
