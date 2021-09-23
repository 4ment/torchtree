from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.linalg

from ..core.abstractparameter import AbstractParameter
from ..core.model import Model
from ..core.parameter import Parameter
from ..core.utils import process_object, register_class
from ..typing import ID


class SubstitutionModel(Model):
    _tag = 'substitution_model'

    def __init__(self, id_: ID, frequencies: AbstractParameter) -> None:
        super().__init__(id_)
        self._frequencies = frequencies

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies.tensor

    @abstractmethod
    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def q(self) -> torch.Tensor:
        pass

    @staticmethod
    def norm(Q, frequencies: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.diagonal(Q, dim1=-2, dim2=-1) * frequencies, -1)


@register_class
class JC69(SubstitutionModel):
    def __init__(self, id_: ID) -> None:
        frequencies = Parameter(None, torch.full((4,), 0.25))
        super().__init__(id_, frequencies)

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


class SymmetricSubstitutionModel(SubstitutionModel, ABC):
    def __init__(self, id_: ID, frequencies: AbstractParameter):
        super().__init__(id_, frequencies)

    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        Q_unnorm = self.q()
        Q = Q_unnorm / SubstitutionModel.norm(Q_unnorm, self.frequencies).unsqueeze(
            -1
        ).unsqueeze(-1)
        sqrt_pi = self.frequencies.sqrt().diag_embed(dim1=-2, dim2=-1)
        sqrt_pi_inv = (1.0 / self.frequencies.sqrt()).diag_embed(dim1=-2, dim2=-1)
        S = sqrt_pi @ Q @ sqrt_pi_inv
        e, v = self.eigen(S)
        offset = branch_lengths.dim() - e.dim() + 1
        return (
            (sqrt_pi_inv @ v).reshape(
                e.shape[:-1] + (1,) * offset + sqrt_pi_inv.shape[-2:]
            )
            @ torch.exp(
                e.reshape(e.shape[:-1] + (1,) * offset + e.shape[-1:])
                * branch_lengths.unsqueeze(-1)
            ).diag_embed()
            @ (v.inverse() @ sqrt_pi).reshape(
                e.shape[:-1] + (1,) * offset + sqrt_pi_inv.shape[-2:]
            )
        )

    def eigen(self, Q: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigh(Q)

    @property
    def sample_shape(self) -> torch.Size:
        return max(
            [parameter.shape[:-1] for parameter in self._parameters.values()],
            key=len,
        )


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


class GeneralSymmetricSubstitutionModel(SymmetricSubstitutionModel):
    def __init__(
        self,
        id_: ID,
        mapping: AbstractParameter,
        rates: AbstractParameter,
        frequencies: AbstractParameter,
    ) -> None:
        super().__init__(id_, frequencies)
        self._rates = rates
        self.mapping = mapping
        self.state_count = frequencies.shape[0]

    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self) -> torch.Tensor:
        indices = torch.triu_indices(self.state_count, self.state_count, 1)
        R = torch.zeros((self.state_count, self.state_count), dtype=self.rates.dtype)
        R[indices[0], indices[1]] = self.rates[self.mapping.tensor]
        R[indices[1], indices[0]] = self.rates[self.mapping.tensor]
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
