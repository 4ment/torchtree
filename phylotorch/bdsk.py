import torch
from torch.distributions.distribution import Distribution


class BDSKY(Distribution):

    def __init__(self, R, delta, s, rho, sampling_times, times, origin, survival=False, validate_args=None):
        """

        :param R: effective reproductive number
        :param delta: total rate of becoming non infectious
        :param s: probability of an individual being sampled
        :param rho: probability of an individual being sampled at present
        :param sampling_times:
        :param times:
        :param origin:
        :param survival:
        :param validate_args:
        """
        self.R = R
        self.delta = delta
        self.s = s
        self.rho = rho
        self.times = times
        self.origin = origin
        self.sampling_times = sampling_times
        self.taxon_count = sampling_times.shape
        self.survival = survival
        batch_shape, event_shape = self.delta.shape[:-1], self.delta.shape[-1:]
        super(BDSKY, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, heights):

        lamb = self.R * self.delta
        psi = self.s * self.delta
        mu = self.delta - psi
        m = mu.shape[0]

        A = torch.sqrt(torch.pow(lamb - mu - psi, 2.0) + 4.0 * lamb * psi)
        B = torch.zeros_like(mu)
        p = torch.ones(m + 1, dtype=mu.dtype)

        sum_term = lamb + mu + psi

        if self.times is None:
            dtimes = (self.origin[0] / self.R.shape[0]).repeat(self.R.shape[0])
            self.times = torch.cat((torch.zeros(1, dtype=dtimes.dtype), dtimes)).cumsum(0)
        else:
            dtimes = self.times[1:] - self.times[:-1]  # t_{i+1} - t_i
        times = self.times[1:]  # does not include 0

        exp_A_term = torch.exp(A * dtimes)
        inv_2lambda = 1.0 / (2. * lamb)

        for i in torch.arange(m - 1, -1, step=-1):
            B[i] += (((1. - 2. * (1. - self.rho[i]) * p[i + 1].clone()) * lamb[i] + mu[i] + psi[i]) / A[i])
            term = exp_A_term[i] * (1. + B[i])
            one_minus_Bi = 1. - B[i]
            p[i] *= ((sum_term[i] - A[i] * (term - one_minus_Bi) / (term + one_minus_Bi)) * inv_2lambda[i])

        # heights are backward and x should be forward in time (i.e. t_0=0)
        x = self.times[-1] - heights
        y = self.times[-1] - self.sampling_times

        # first term
        index = torch.min(self.times > self.origin[0] + heights[-1], 0)[1]
        e = torch.exp(A[0] * dtimes[0])
        q0 = 4. * e / torch.pow(e * (1. + B[0]) + (1. - B[0]), 2)

        # condition on sampling at least one individual
        if self.survival:
            term1 = torch.log(q0 / (1. - p[0]))
        else:
            term1 = torch.log(q0)

        indices_x = torch.clamp(torch.min(self.times > x.unsqueeze(-1), 1)[1], max=lamb.shape[0] - 1)
        e = torch.exp(-A[indices_x] * (x - times[indices_x]))
        term2 = torch.log(lamb[indices_x]) + torch.log(
            4. * e / torch.pow(e * (1. + B[indices_x]) + (1. - B[indices_x]), 2))

        # serially sampled term
        if torch.all(self.s != 0.):
            indices_y = torch.clamp(torch.min(self.times > y.unsqueeze(-1), 1)[1], max=lamb.shape[0] - 1)
            e = torch.exp(-A[indices_y] * (y - self.times[indices_y]))
            term3 = torch.log(psi[indices_y]) - torch.log(
                4. * e / torch.pow(e * (1. + B[indices_y]) + (1. - B[indices_y]), 2))
        else:
            term3 = torch.zeros(1)

        # last term
        # n = torch.stack([torch.sum(x > times[i]) - torch.sum(y >= times[i]) for i in range(m)])
        if m == 1:
            n = torch.zeros(1, dtype=torch.int64)
        else:
            n = torch.cat([torch.zeros(1, dtype=torch.int64),
                           1 + torch.stack([torch.sum(heights > (self.times[-1] - self.times[i])) - torch.sum(
                               self.sampling_times >= (self.times[-1] - self.times[i])) for i in range(1, m)])])

        # contemporenaous term
        if torch.all(self.s == 0.):
            term4 = (n * torch.log(4. * exp_A_term / torch.pow(exp_A_term * (1 + B) + (1 - B), 2))).sum() + \
                    self.sampling_times.shape[0] * torch.log(self.rho[-1:])[0]
            # torch.where(self.rho > 0., self.sampling_times.shape[0] * torch.log(self.rho), self.rho)
        else:
            term4 = torch.zeros(1)
        return term1 + term2.sum() + term3.sum() + term4
