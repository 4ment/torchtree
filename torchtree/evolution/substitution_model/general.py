from __future__ import annotations

from typing import Optional, Union

import torch
import torch.linalg
from torch import Tensor

from ...core.abstractparameter import AbstractParameter
from ...core.parameter import Parameter
from ...core.utils import process_object, register_class
from ...evolution.datatype import DataType
from ...typing import ID
from .abstract import (
    NonSymmetricSubstitutionModel,
    SubstitutionModel,
    SymmetricSubstitutionModel,
)


@register_class
class GeneralJC69(SubstitutionModel):
    def __init__(self, id_: ID, state_count: int) -> None:
        super().__init__(id_)
        self._frequencies = torch.full((state_count,), 1.0 / state_count)
        self.state_count = state_count

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies

    @property
    def rates(self) -> Union[Tensor, list[Tensor]]:
        return []

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def _sample_shape(self) -> torch.Size:
        return torch.Size([])

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._frequencies.cuda(device)

    def cpu(self) -> None:
        self._frequencies.cpu()

    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        d = torch.unsqueeze(branch_lengths, -1)
        a = 1.0 / self.state_count + (
            self.state_count - 1.0
        ) / self.state_count * torch.exp(
            -self.state_count / (self.state_count - 1.0) * d
        )
        b = (
            1.0 / self.state_count
            - torch.exp(-self.state_count / (self.state_count - 1.0) * d)
            / self.state_count
        )
        P = b.unsqueeze(-1).repeat(
            (1,) * branch_lengths.dim() + (self.state_count, self.state_count)
        )
        P[..., range(self.state_count), range(self.state_count)] = a.repeat(
            (1,) * branch_lengths.dim() + (self.state_count,)
        )
        return P

    def q(self) -> torch.Tensor:
        Q = torch.full(
            (self.state_count, self.state_count),
            1.0 / (self.state_count - 1),
            dtype=self._frequencies.dtype,
        )
        Q[range(self.state_count), range(self.state_count)] = -1.0
        return Q

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        state_count = data['state_count']
        return cls(id_, state_count)


@register_class
class GeneralSymmetricSubstitutionModel(SymmetricSubstitutionModel):
    def __init__(
        self,
        id_: ID,
        data_type: DataType,
        mapping: AbstractParameter,
        rates: AbstractParameter,
        frequencies: AbstractParameter,
    ) -> None:
        super().__init__(id_, frequencies)
        self._rates = rates
        self.mapping = mapping
        self.state_count = data_type.state_count
        self.data_type = data_type

    @property
    def rates(self) -> Union[Tensor, list[Tensor]]:
        return self._rates.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self) -> torch.Tensor:
        indices = torch.triu_indices(self.state_count, self.state_count, 1)
        R = torch.zeros(
            self._rates.tensor.shape[:-1] + (self.state_count, self.state_count),
            dtype=self.rates.dtype,
        )
        R[..., indices[0], indices[1]] = self.rates[..., self.mapping.tensor]
        R[..., indices[1], indices[0]] = self.rates[..., self.mapping.tensor]
        identity = torch.eye(self.state_count)
        for _ in range(R.dim() - 2):
            identity = identity.unsqueeze(0)
        pi = self.frequencies.unsqueeze(-1) * identity.repeat(R.shape[:-2] + (1, 1))
        Q = R @ pi
        Q[..., range(self.state_count), range(self.state_count)] = -torch.sum(Q, dim=-1)
        return Q

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        data_type = process_object(data['data_type'], dic)
        rates = process_object(data['rates'], dic)
        frequencies = process_object(data['frequencies'], dic)
        mapping = process_object(data['mapping'], dic)
        return cls(id_, data_type, mapping, rates, frequencies)


@register_class
class GeneralNonSymmetricSubstitutionModel(NonSymmetricSubstitutionModel):
    def __init__(
        self,
        id_: ID,
        data_type: DataType,
        mapping: AbstractParameter,
        rates: AbstractParameter,
        frequencies: AbstractParameter,
        normalize: bool,
    ) -> None:
        super().__init__(id_, frequencies)
        self._rates = rates
        self.mapping = mapping
        self.state_count = data_type.state_count
        self.data_type = data_type
        self.normalize = normalize

    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self) -> torch.Tensor:
        indices = torch.triu_indices(self.state_count, self.state_count, 1)
        R = torch.zeros(
            self._rates.tensor.shape[:-1] + (self.state_count, self.state_count),
            dtype=self._rates.dtype,
        )
        dim = int(self.mapping.shape[-1] / 2)
        R[..., indices[0], indices[1]] = self.rates[..., self.mapping.tensor[:dim]]
        R[..., indices[1], indices[0]] = self.rates[..., self.mapping.tensor[dim:]]
        identity = torch.eye(self.state_count)
        for _ in range(R.dim() - 2):
            identity = identity.unsqueeze(0)
        pi = self.frequencies.unsqueeze(-1) * identity.repeat(R.shape[:-2] + (1, 1))
        Q = R @ pi
        Q[..., range(self.state_count), range(self.state_count)] = -torch.sum(Q, dim=-1)
        return Q

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        data_type = process_object(data['data_type'], dic)
        rates = process_object(data['rates'], dic)
        frequencies = process_object(data['frequencies'], dic)
        if isinstance(data['mapping'], list):
            mapping = Parameter(None, torch.tensor(data['mapping']))
        else:
            mapping = process_object(data['mapping'], dic)
        normalize = data.get('normalize', True)
        return cls(id_, data_type, mapping, rates, frequencies, normalize)


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

    def _sample_shape(self) -> torch.Size:
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
