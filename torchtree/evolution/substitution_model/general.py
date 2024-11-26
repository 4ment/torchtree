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
    r"""General symmetric substitution model.

    The state space :math:`\Omega=\{S_0, S_1, \dots, S_{M-1}\}` of this model is defined by the `DataType` object.
    
    This model is composed of:

    - :math:`K` substitution rate parameters: :math:`\mathbf{r}=r_0, r_1, \dots, r_{K-1}` where :math:`K \leq (M^2-M)/2`.
    - :math:`M` equilibrium frequency parameters: :math:`\pi_0, \pi_1, \dots, \pi_{M-1}`.
    - A mapping function that associates each matrix element :math:`Q_{ij}` to an index in the set of rates :math:`f: \{0, 1, \dots, (M^2-M)/2-1\} \rightarrow \{0,1, \dots, K-1\}`
    
    The matrix :math:`Q` is thus defined as:

    .. math::

        Q_{ij} =
        \begin{cases}
        r_{f(i \cdot M + j)} \pi_j & \text{if } i \neq j \\
        -\sum_{k \neq i} Q_{ik} & \text{if } i = j
        \end{cases}
    
    where :math:`i,j \in \{0,1, \dots, M-1\}` are zero-based indices for rows and columns.

    :math:`f` is implemented as a one-dimentional array :math:`\mathbf{g}[x]=f(x)` for :math:`x \in \{0, 1,\dots, (M^2-M)/2-1\}` where each element maps a position in :math:`Q` to an index in the rate array :math:`r`.
    The mapping is defined such as the position :math:`(i,j)` in :math:`Q` corresponds to :math:`i \cdot M+ j` for :math:`i \neq j`.
    The indices correspond to first iterating over rows (row 0, then row 1, etc.) and then over columns for each row of the upper off-diagonal elements.

    The HKY substitution model can be defined as a symmetric substitution model with M=4 frequency parameters and rate parameters :math:`\mathbf{r}=r_0, r_1`.
    The mapping function is therefore:
    
    .. math::

        f(k) =
        \begin{cases}
        0 & \text{if } k = i \cdot 4 + j \text{ and } i \rightarrow j \text{ is transversion}\\
        1 & \text{otherwise}
        \end{cases}
    
    As a one-dimentional array, the mapping is defined as :math:`\mathbf{g}=[0,1,0,0,1,0]`.
    
    The HKY rate matrix :math:`Q` is given as:

    .. math::

        Q_{HKY} = 
        \begin{bmatrix}
        -(r_0 \pi_C + r_1 \pi_G + r_0 \pi_T) & r_0 \pi_C & r_1 \pi_G & r_0 \pi_T \\
        r_0 \pi_A & -(r_0 \pi_A + r_0 \pi_G + r_0 \pi_T) & r_0 \pi_G & r_1 \pi_T \\
        r_1 \pi_A & r_0 \pi_C & -(r_1\pi_A + r_0 \pi_C + r_0 \pi_T) & r_0 \pi_T \\
        r_0 \pi_A & r_1 \pi_C & r_0 \pi_G & -(r_0 \pi_A + r_1 \pi_C + r_0 \pi_G)
        \end{bmatrix}
    
    Similarly the GTR model can be specified with :math:`\mathbf{g}=[0,1,2,3,4,5]` and :math:`\mathbf{r}=r_0, r_1, r_2, r_3, r_4, r_5`.
    
    .. note::
        The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be the order of the states defined in the DataType object.
    """

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
        if 'mapping' not in data:
            mapping_count = data_type.state_count * (data_type.state_count - 1) // 2
            mapping = Parameter(None, torch.arange(mapping_count))
        elif isinstance(data['mapping'], list):
            mapping = Parameter(None, torch.tensor(data['mapping']))
        else:
            mapping = process_object(data['mapping'], dic)
        return cls(id_, data_type, mapping, rates, frequencies)


@register_class
class GeneralNonSymmetricSubstitutionModel(NonSymmetricSubstitutionModel):
    r"""General non-symmetric substitution model.

    The state space :math:`\Omega=\{S_0, S_1, \dots, S_{M-1}\}` of this model is defined by the `DataType` object.
    
    This model is composed of:

    - :math:`K` substitution rate parameters: :math:`\mathbf{r}=r_0, r_1, \dots, r_{K-1}` where :math:`K \leq (M^2-M)`.
    - :math:`M` equilibrium frequency parameters: :math:`\pi_0, \pi_1, \dots, \pi_{M-1}`.
    - A mapping function that associates each matrix element :math:`Q_{ij}` to an index in the set of rates :math:`f: \{0, 1, \dots, (M^2-M)-1\} \rightarrow \{0,1, \dots, K-1\}`
    
    The matrix :math:`Q` is thus defined as:

    .. math::

        Q_{ij} =
        \begin{cases}
        r_{f(i \cdot M + j)} \pi_j & \text{if } i \neq j \\
        -\sum_{k \neq i} Q_{ik} & \text{if } i = j
        \end{cases}
    
    where :math:`i,j \in \{0,1, \dots, M-1\}` are zero-based indices for rows and columns.

    :math:`f` is implemented as a one-dimentional array :math:`\mathbf{g}[x]=f(x)` for :math:`x \in \{0, 1,\dots, (M^2-M)-1\}` where each element maps a position in :math:`Q` to an index in the rate array :math:`r`.
    The mapping is defined such as the position :math:`(i,j)` in :math:`Q` corresponds to :math:`i \cdot M + j` for :math:`i > j` and :math:`j \cdot M + i + (M^2-M)/2` for :math:`i < j`.
    In other words, the first of :math:`\mathbf{g}` corresponds to the upper off-diagonal elements and the second to the lower off-diagonal elements.
    
    .. note::
        The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be the order of the states defined in the DataType object.
    """

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
        if 'mapping' not in data:
            mapping_count = data_type.state_count * (data_type.state_count - 1)
            mapping = Parameter(None, torch.arange(mapping_count))
        elif isinstance(data['mapping'], list):
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
