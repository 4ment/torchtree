import torch
import torch.linalg

from ...core.abstractparameter import AbstractParameter
from ...core.utils import process_object, register_class
from ...typing import ID
from .abstract import SubstitutionModel, SymmetricSubstitutionModel


@register_class
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


@register_class
class EmpiricalSubstitutionModel(SubstitutionModel):
    def __init__(self, id_: ID, rates: torch.Tensor, frequencies: torch.Tensor):
        super().__init__(id_)
        self._rates = rates
        self._frequencies = frequencies
        self.Q = self.create_rate_matrix(rates, frequencies)
        self.sqrt_pi = self.frequencies.sqrt().diag_embed(dim1=-2, dim2=-1)
        self.sqrt_pi_inv = (1.0 / self.frequencies.sqrt()).diag_embed(dim1=-2, dim2=-1)
        Q = self.Q / -torch.sum(
            torch.diagonal(self.Q, dim1=-2, dim2=-1) * self.frequencies, -1
        ).unsqueeze(-1).unsqueeze(-1)
        self.e, self.v = self.eigen(self.sqrt_pi @ Q @ self.sqrt_pi_inv)

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies

    @property
    def sample_shape(self) -> torch.Size:
        return torch.Size([])

    def q(self) -> torch.Tensor:
        return self.Q

    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        offset = branch_lengths.dim() - self.e.dim() + 1
        return (
            (self.sqrt_pi_inv @ self.v).reshape(
                self.e.shape[:-1] + (1,) * offset + self.sqrt_pi_inv.shape[-2:]
            )
            @ torch.exp(
                self.e.reshape(self.e.shape[:-1] + (1,) * offset + self.e.shape[-1:])
                * branch_lengths.unsqueeze(-1)
            ).diag_embed()
            @ (self.v.inverse() @ self.sqrt_pi).reshape(
                self.e.shape[:-1] + (1,) * offset + self.sqrt_pi_inv.shape[-2:]
            )
        )

    def eigen(self, Q: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigh(Q)

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    def handle_parameter_changed(
        self, variable: AbstractParameter, index, event
    ) -> None:
        pass

    @staticmethod
    def create_rate_matrix(
        rates: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        state_count = frequencies.shape[-1]
        R = torch.zeros((state_count, state_count), dtype=rates.dtype)
        tril_indices = torch.triu_indices(row=state_count, col=state_count, offset=1)
        R[tril_indices[0], tril_indices[1]] = rates
        R[tril_indices[1], tril_indices[0]] = rates
        Q = R @ frequencies.diag()
        Q[range(state_count), range(state_count)] = -torch.sum(Q, dim=1)
        return Q

    @classmethod
    def from_json(cls, data, dic):
        rates = data['rates']
        frequencies = data['frequencies']
        return cls(data['id'], rates, frequencies)
