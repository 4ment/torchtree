from abc import abstractmethod

import numpy as np
import torch


class SubstitutionModel(object):
    def __init__(self, frequencies):
        self.frequencies = frequencies

    @abstractmethod
    def p_t(self, branch_lengths):
        pass

    @abstractmethod
    def q(self):
        pass

    @staticmethod
    def norm(Q, frequencies):
        return -torch.sum(torch.diagonal(Q) * frequencies)


class JC69(SubstitutionModel):
    def __init__(self):
        SubstitutionModel.__init__(self, torch.tensor(np.array([0.25] * 4)))

    def p_t(self, branch_lengths):
        """Calculate transition probability matrices

        :param branch_lengths: tensor of branch lengths [B,K]
        :return: tensor of probability matrices [B,K,4,4]
        """
        d = torch.unsqueeze(branch_lengths, -1)
        a = 0.25 + 3. / 4. * torch.exp(-4. / 3. * d)
        b = 0.25 - 0.25 * torch.exp(-4. / 3. * d)
        return torch.cat((a, b, b, b,
                          b, a, b, b,
                          b, b, a, b,
                          b, b, b, a), -1).reshape(d.shape[0], d.shape[1], 4, 4)

    def q(self):
        return torch.tensor(np.array([[-1., 1. / 3, 1. / 3, 1. / 3],
                                      [1. / 3, -1., 1. / 3, 1. / 3],
                                      [1. / 3, 1. / 3, -1., 1. / 3],
                                      [1. / 3, 1. / 3, 1. / 3, -1.]]))


class SymmetricSubstitutionModel(SubstitutionModel):
    def __init__(self, frequencies):
        SubstitutionModel.__init__(self, frequencies)

    def p_t(self, branch_lengths):
        Q_unnorm = self.q()
        Q = Q_unnorm / SubstitutionModel.norm(Q_unnorm, self.frequencies)
        sqrt_pi = torch.zeros(Q.shape, dtype=torch.float64)
        sqrt_pi_inv = torch.zeros(Q.shape, dtype=torch.float64)
        sqrt_pi[range(sqrt_pi.shape[0]), range(sqrt_pi.shape[0])] = self.frequencies.sqrt()
        sqrt_pi_inv[range(sqrt_pi_inv.shape[0]), range(sqrt_pi_inv.shape[0])] = 1. / self.frequencies.sqrt()
        S = sqrt_pi @ Q @ sqrt_pi_inv
        e, v = self.eigen(S)
        return sqrt_pi_inv @ v @ torch.exp(e * branch_lengths).diag_embed() @ v.inverse() @ sqrt_pi

    def eigen(self, Q):
        return torch.symeig(Q, eigenvectors=True)


class GTR(SymmetricSubstitutionModel):
    def __init__(self, rates, frequencies):
        SymmetricSubstitutionModel.__init__(self, frequencies)
        self.rates = rates

    def q(self):
        rates = self.rates
        pi = self.frequencies
        return torch.stack((
            -(rates[0] * pi[1] + rates[1] * pi[2] + rates[2] * pi[3]),
            rates[0] * pi[1],
            rates[1] * pi[2],
            rates[2] * pi[3],

            rates[0] * pi[0],
            -(rates[0] * pi[0] + rates[3] * pi[2] + rates[4] * pi[3]),
            rates[3] * pi[2],
            rates[4] * pi[3],

            rates[1] * pi[0],
            rates[3] * pi[1],
            -(rates[1] * pi[0] + rates[3] * pi[1] + rates[5] * pi[3]),
            rates[5] * pi[3],

            rates[2] * pi[0],
            rates[4] * pi[1],
            rates[5] * pi[2],
            -(rates[2] * pi[0] + rates[4] * pi[1] + rates[5] * pi[2])), 0).reshape((4, 4))


class HKY(SymmetricSubstitutionModel):
    def __init__(self, kappa, frequencies):
        SymmetricSubstitutionModel.__init__(self, frequencies)
        self.kappa = kappa

    def p_t(self, branch_lengths):
        d = torch.unsqueeze(branch_lengths, -1)
        pi = self.frequencies
        R = pi[0] + pi[2]
        Y = pi[3] + pi[1]
        kappa = self.kappa[0]
        k1 = kappa * Y + R
        k2 = kappa * R + Y
        r = 1. / (2. * (pi[0] * pi[1] + pi[1] * pi[2] + pi[0] * pi[3] + pi[2] * pi[3] + kappa * (
                pi[1] * pi[3] + pi[0] * pi[2])))

        exp1 = torch.exp(-d * r)
        exp22 = torch.exp(-k2 * d * r)
        exp21 = torch.exp(-k1 * d * r)
        return torch.cat((pi[0] * (1. + (Y / R) * exp1) + (pi[2] / R) * exp22,
                          pi[1] * (1. - exp1),
                          pi[2] * (1. + (Y / R) * exp1) - (pi[2] / R) * exp22,
                          pi[3] * (1. - exp1),

                          pi[0] * (1. - exp1),
                          pi[1] * (1. + (R / Y) * exp1) + (pi[3] / Y) * exp21,
                          pi[2] * (1. - exp1),
                          pi[3] * (1. + (R / Y) * exp1) - (pi[3] / Y) * exp21,

                          pi[0] * (1. + (Y / R) * exp1) - (pi[0] / R) * exp22,
                          pi[1] * (1. - exp1),
                          pi[2] * (1. + (Y / R) * exp1) + (pi[0] / R) * exp22,
                          pi[3] * (1. - exp1),

                          pi[0] * (1. - exp1),
                          pi[1] * (1. + (R / Y) * exp1) - (pi[1] / Y) * exp21,
                          pi[2] * (1. - exp1),
                          pi[3] * (1. + (R / Y) * exp1) + (pi[1] / Y) * exp21)).reshape(
            d.shape[0], d.shape[1], 4, 4)

    def q(self):
        pi = self.frequencies.unsqueeze(-1)
        return torch.cat((-(pi[1] + self.kappa * pi[2] + pi[3]), pi[1], self.kappa * pi[2], pi[3],
                          pi[0], -(pi[0] + pi[2] + self.kappa * pi[3]), pi[2], self.kappa * pi[3],
                          self.kappa * pi[0], pi[1], -(self.kappa * pi[0] + pi[1] + pi[3]), pi[3],
                          pi[0], self.kappa * pi[1], pi[2], -(pi[0] + self.kappa * pi[1] + pi[2])), 0).reshape((4, 4))
