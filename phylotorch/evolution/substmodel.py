from abc import abstractmethod

import numpy as np
import torch

from ..core.model import Model, Parameter
from ..core.utils import process_object


class SubstitutionModel(Model):
    def __init__(self, id_, frequencies):
        self._frequencies = frequencies
        super(SubstitutionModel, self).__init__(id_)

    @property
    def frequencies(self):
        return self._frequencies.tensor

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
    def __init__(self, id_):
        frequencies = Parameter('frequencies', torch.tensor(np.repeat(0.25, 4)))
        super(JC69, self).__init__(id_, frequencies)

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

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        return cls(data['id'])


class SymmetricSubstitutionModel(SubstitutionModel):

    def __init__(self, id_, frequencies):
        self.add_parameter(frequencies)
        super(SymmetricSubstitutionModel, self).__init__(id_, frequencies)

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
    def __init__(self, id_, rates, frequencies):
        self._rates = rates
        self.add_parameter(rates)
        super(GTR, self).__init__(id_, frequencies)

    @property
    def rates(self):
        return self._rates.tensor

    def update(self, value):
        if isinstance(value, dict):
            if self._rates.id in value:
                self._rates.tensor = value[self._rates.id]
            if self._frequencies.id in value:
                self._frequencies.tensor = value[self._frequencies.id]
        else:
            self._rates, self._frequencies = value

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

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

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        rates = process_object(data['rates'], dic)
        frequencies = process_object(data['frequencies'], dic)
        return cls(id_, rates, frequencies)


class HKY(SymmetricSubstitutionModel):
    def __init__(self, id_, kappa, frequencies):
        self._kappa = kappa
        self.add_parameter(kappa)
        super(SymmetricSubstitutionModel, self).__init__(id_, frequencies)

    @property
    def kappa(self):
        return self._kappa.tensor

    def update(self, value):
        if isinstance(value, dict):
            if self._kappa.id in value:
                self._kappa.tensor = value[self._kappa.id]
            if self._frequencies.id in value:
                self._frequencies.tensor = value[self._frequencies.id]
        else:
            self._kappa, self._frequencies = value

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def p_t2(self, branch_lengths):
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
                          pi[3] * (1. + (R / Y) * exp1) + (pi[1] / Y) * exp21), -1).reshape(
            d.shape[0], d.shape[1], 4, 4)

    def q(self):
        pi = self.frequencies.unsqueeze(-1)
        return torch.cat((-(pi[1] + self.kappa * pi[2] + pi[3]), pi[1], self.kappa * pi[2], pi[3],
                          pi[0], -(pi[0] + pi[2] + self.kappa * pi[3]), pi[2], self.kappa * pi[3],
                          self.kappa * pi[0], pi[1], -(self.kappa * pi[0] + pi[1] + pi[3]), pi[3],
                          pi[0], self.kappa * pi[1], pi[2], -(pi[0] + self.kappa * pi[1] + pi[2])), 0).reshape((4, 4))

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        rates = process_object(data['kappa'], dic)
        frequencies = process_object(data['frequencies'], dic)
        return cls(id_, rates, frequencies)


class GeneralSymmetricSubstitutionModel(SymmetricSubstitutionModel):

    def __init__(self, id_, mapping, rates, frequencies):
        self._rates = rates
        self.mapping = mapping
        self.state_count = frequencies.shape[0]
        self.add_parameter(rates)
        super(GeneralSymmetricSubstitutionModel, self).__init__(id_, frequencies)

    @property
    def rates(self):
        return self._rates.tensor

    def update(self, value):
        if isinstance(value, dict):
            if self._rates.id in value:
                self._rates.tensor = value[self._rates.id]
            if self._frequencies.id in value:
                self._frequencies.tensor = value[self._frequencies.id]
        else:
            self._rates, self._frequencies = value

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self):
        indices = torch.triu_indices(self.state_count, self.state_count, 1)
        R = torch.zeros((self.state_count, self.state_count), dtype=self.rates.dtype)
        R[indices[0], indices[1]] = self.rates
        R[indices[1], indices[0]] = self.rates
        Q = R @ self.frequencies.diag()
        Q[range(len(Q)), range(len(Q))] = -torch.sum(Q, dim=1)
        return Q

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        rates = process_object(data['rates'], dic)
        frequencies = process_object(data['frequencies'], dic)
        mapping = process_object(data['mapping'], dic)
        return cls(id_, mapping, rates, frequencies)
