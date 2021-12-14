from typing import Optional, Union

import torch
import torch.linalg

from ...core.abstractparameter import AbstractParameter
from ...core.utils import process_object, register_class
from ...typing import ID
from .abstract import SubstitutionModel, SymmetricSubstitutionModel


@register_class
class JC69(SubstitutionModel):
    def __init__(self, id_: ID) -> None:
        super().__init__(id_)
        self._frequencies = torch.full((4,), 0.25)

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies

    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate transition probability matrices.

        :param branch_lengths: tensor of branch lengths [B,K]
        :return: tensor of probability matrices [B,K,4,4]
        """
        d = torch.unsqueeze(branch_lengths, -1)
        a = 0.25 + 3.0 / 4.0 * torch.exp(-4.0 / 3.0 * d)
        b = 0.25 - 0.25 * torch.exp(-4.0 / 3.0 * d)
        return torch.cat((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
            d.shape[:-1] + (4, 4)
        )

    def q(self) -> torch.Tensor:
        return torch.tensor(
            [
                [-1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [1.0 / 3, -1.0, 1.0 / 3, 1.0 / 3],
                [1.0 / 3, 1.0 / 3, -1.0, 1.0 / 3],
                [1.0 / 3, 1.0 / 3, 1.0 / 3, -1.0],
            ],
            dtype=self.frequencies.dtype,
            device=self.frequencies.device,
        )

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return torch.Size([])

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._frequencies.cuda(device)

    def cpu(self) -> None:
        self._frequencies.cpu()

    @classmethod
    def from_json(cls, data, dic):
        return cls(data['id'])


@register_class
class HKY(SymmetricSubstitutionModel):
    def __init__(
        self, id_: ID, kappa: AbstractParameter, frequencies: AbstractParameter
    ) -> None:
        super().__init__(id_, frequencies)
        self._kappa = kappa

    @property
    def kappa(self) -> torch.Tensor:
        return self._kappa.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def p_t_analytical(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        # FIXME: does not work with K>1 rate categories
        raise NotImplementedError
        if len(self.frequencies.shape) == 1:
            pi = self.frequencies.unsqueeze(0)
            kappa = self.kappa.unsqueeze(0)
        else:
            pi = self.frequencies.unsqueeze(-2).unsqueeze(-2)
            kappa = self.kappa.unsqueeze(-1)
        R = pi[..., 0] + pi[..., 2]
        Y = pi[..., 3] + pi[..., 1]
        k1 = kappa * Y + R
        k2 = kappa * R + Y
        r = 1.0 / (
            2.0
            * (
                pi[..., 0] * pi[..., 1]
                + pi[..., 1] * pi[..., 2]
                + pi[..., 0] * pi[..., 3]
                + pi[..., 2] * pi[..., 3]
                + kappa * (pi[..., 1] * pi[..., 3] + pi[..., 0] * pi[..., 2])
            )
        )

        exp1 = torch.exp(-branch_lengths * r)
        exp22 = torch.exp(-k2 * branch_lengths * r)
        exp21 = torch.exp(-k1 * branch_lengths * r)
        return torch.cat(
            (
                pi[..., 0] * (1.0 + (Y / R) * exp1) + (pi[..., 2] / R) * exp22,
                pi[..., 1] * (1.0 - exp1),
                pi[..., 2] * (1.0 + (Y / R) * exp1) - (pi[..., 2] / R) * exp22,
                pi[..., 3] * (1.0 - exp1),
                pi[..., 0] * (1.0 - exp1),
                pi[..., 1] * (1.0 + (R / Y) * exp1) + (pi[..., 3] / Y) * exp21,
                pi[..., 2] * (1.0 - exp1),
                pi[..., 3] * (1.0 + (R / Y) * exp1) - (pi[..., 3] / Y) * exp21,
                pi[..., 0] * (1.0 + (Y / R) * exp1) - (pi[..., 0] / R) * exp22,
                pi[..., 1] * (1.0 - exp1),
                pi[..., 2] * (1.0 + (Y / R) * exp1) + (pi[..., 0] / R) * exp22,
                pi[..., 3] * (1.0 - exp1),
                pi[..., 0] * (1.0 - exp1),
                pi[..., 1] * (1.0 + (R / Y) * exp1) - (pi[..., 1] / Y) * exp21,
                pi[..., 2] * (1.0 - exp1),
                pi[..., 3] * (1.0 + (R / Y) * exp1) + (pi[..., 1] / Y) * exp21,
            ),
            -1,
        ).reshape(branch_lengths.shape + (4, 4))

    def q(self) -> torch.Tensor:
        if len(self.frequencies.shape) == 1:
            pi = self.frequencies.unsqueeze(0)
        else:
            pi = self.frequencies.unsqueeze(-2)
        kappa = self.kappa
        return torch.cat(
            (
                -(pi[..., 1] + kappa * pi[..., 2] + pi[..., 3]),
                pi[..., 1],
                kappa * pi[..., 2],
                pi[..., 3],
                pi[..., 0],
                -(pi[..., 0] + pi[..., 2] + kappa * pi[..., 3]),
                pi[..., 2],
                kappa * pi[..., 3],
                kappa * pi[..., 0],
                pi[..., 1],
                -(kappa * pi[..., 0] + pi[..., 1] + pi[..., 3]),
                pi[..., 3],
                pi[..., 0],
                kappa * pi[..., 1],
                pi[..., 2],
                -(pi[..., 0] + kappa * pi[..., 1] + pi[..., 2]),
            ),
            -1,
        ).reshape(kappa.shape[:-1] + (4, 4))

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        rates = process_object(data['kappa'], dic)
        frequencies = process_object(data['frequencies'], dic)
        return cls(id_, rates, frequencies)


@register_class
class GTR(SymmetricSubstitutionModel):
    def __init__(
        self, id_: ID, rates: AbstractParameter, frequencies: AbstractParameter
    ):
        super().__init__(id_, frequencies)
        self._rates = rates

    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self) -> torch.Tensor:
        if len(self.frequencies.shape) == 1:
            pi = self.frequencies.unsqueeze(0)
            rates = self.rates.unsqueeze(0)
        else:
            pi = self.frequencies.unsqueeze(-2)
            rates = self.rates.unsqueeze(-2)
        return torch.cat(
            (
                -(
                    rates[..., 0] * pi[..., 1]
                    + rates[..., 1] * pi[..., 2]
                    + rates[..., 2] * pi[..., 3]
                ),
                rates[..., 0] * pi[..., 1],
                rates[..., 1] * pi[..., 2],
                rates[..., 2] * pi[..., 3],
                rates[..., 0] * pi[..., 0],
                -(
                    rates[..., 0] * pi[..., 0]
                    + rates[..., 3] * pi[..., 2]
                    + rates[..., 4] * pi[..., 3]
                ),
                rates[..., 3] * pi[..., 2],
                rates[..., 4] * pi[..., 3],
                rates[..., 1] * pi[..., 0],
                rates[..., 3] * pi[..., 1],
                -(
                    rates[..., 1] * pi[..., 0]
                    + rates[..., 3] * pi[..., 1]
                    + rates[..., 5] * pi[..., 3]
                ),
                rates[..., 5] * pi[..., 3],
                rates[..., 2] * pi[..., 0],
                rates[..., 4] * pi[..., 1],
                rates[..., 5] * pi[..., 2],
                -(
                    rates[..., 2] * pi[..., 0]
                    + rates[..., 4] * pi[..., 1]
                    + rates[..., 5] * pi[..., 2]
                ),
            ),
            -1,
        ).reshape(self.rates.shape[:-1] + (4, 4))

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        rates = process_object(data['rates'], dic)
        frequencies = process_object(data['frequencies'], dic)
        return cls(id_, rates, frequencies)
