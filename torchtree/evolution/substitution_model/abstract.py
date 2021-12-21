from abc import ABC, abstractmethod

import torch
import torch.linalg

from ...core.abstractparameter import AbstractParameter
from ...core.model import Model
from ...typing import ID


class SubstitutionModel(Model):
    _tag = 'substitution_model'

    def __init__(self, id_: ID) -> None:
        super().__init__(id_)

    @property
    @abstractmethod
    def frequencies(self) -> torch.Tensor:
        pass

    @abstractmethod
    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def q(self) -> torch.Tensor:
        pass


class AbstractSubstitutionModel(SubstitutionModel, ABC):
    def __init__(self, id_: ID, frequencies: AbstractParameter) -> None:
        super().__init__(id_)
        self._frequencies = frequencies

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies.tensor

    def norm(self, Q) -> torch.Tensor:
        return -torch.sum(torch.diagonal(Q, dim1=-2, dim2=-1) * self.frequencies, -1)


class SymmetricSubstitutionModel(AbstractSubstitutionModel, ABC):
    def __init__(self, id_: ID, frequencies: AbstractParameter):
        super().__init__(id_, frequencies)

    def p_t(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        Q_unnorm = self.q()
        Q = Q_unnorm / self.norm(Q_unnorm).unsqueeze(-1).unsqueeze(-1)
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
