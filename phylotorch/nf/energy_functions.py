import math

import torch

from phylotorch.core.model import CallableModel
from phylotorch.core.utils import process_object


class EnergyFunctionModel(CallableModel):

    def __init__(self, id_, x, desc):
        super(EnergyFunctionModel, self).__init__(id_)
        self.x = x
        self.desc = desc

        w1 = lambda z: torch.sin(2 * math.pi * z[:, 0] / 4)
        w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
        w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

        if self.desc == 'u_z1':
            self.U = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4) ** 2 - \
                               torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + torch.exp(
                                   -0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
        elif self.desc == 'u_z2':
            self.U = lambda z: 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
        elif self.desc == 'u_z3':
            self.U = lambda z: - torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2) + torch.exp(
                -0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) + 1e-10)
        else:
            self.U = lambda z: - torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2) + torch.exp(
                -0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) + 1e-10)

    def log_prob(self, value):
        pass

    def _call(self):
        if self.x.tensor.dim() == 1:
            z = self.x.tensor.unsqueeze(0)
        else:
            z = self.x.tensor
        return -self.U(z)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_object(data['x'], dic)
        return cls(id_, x, data['function'])
