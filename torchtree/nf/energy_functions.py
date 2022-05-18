import math

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import get_class, process_object
from ..typing import ID


def w1(z):
    return torch.sin(2 * math.pi * z[:, 0] / 4)


def w2(z):
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def w3(z):
    return 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)


class EnergyFunctionModel(CallableModel):
    def __init__(
        self, id_: ID, x: AbstractParameter, desc: str, dtype=None, device=None
    ):
        super().__init__(id_)
        self.x = x
        self.desc = desc

        if self.desc == 'u_z1':
            self.U = lambda z: 0.5 * (
                (torch.norm(z, p=2, dim=1) - 2) / 0.4
            ) ** 2 - torch.log(
                torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
            )
        elif self.desc == 'u_z2':
            self.U = lambda z: 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
        elif self.desc == 'u_z3':
            self.U = lambda z: -torch.log(
                torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
                + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
                + 1e-10
            )
        else:
            self.U = lambda z: -torch.log(
                torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
                + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
                + 1e-10
            )
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        if self.x.tensor.dim() == 1:
            z = self.x.tensor.unsqueeze(0)
        else:
            z = self.x.tensor
        return -self.U(z)

    def handle_model_changed(self, model, obj, index):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.x.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        if 'dtype' in data:
            dtype = get_class(data['dtype'])
        else:
            dtype = None
        device = data.get('device', None)
        x = process_object(data['x'], dic)
        return cls(id_, x, data['function'], dtype, device)
