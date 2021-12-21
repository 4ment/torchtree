from itertools import combinations

import numpy
import torch

from ...core.abstractparameter import AbstractParameter
from ...core.utils import process_object, register_class
from ...typing import ID
from ..datatype import CodonDataType
from .abstract import SymmetricSubstitutionModel


@register_class
class MG94(SymmetricSubstitutionModel):
    def __init__(
        self,
        id_: ID,
        data_type: CodonDataType,
        alpha: AbstractParameter,
        beta: AbstractParameter,
        kappa: AbstractParameter,
        frequencies: AbstractParameter,
    ):
        super().__init__(id_, frequencies)
        self.data_type = data_type
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        coding_indices = [
            i for i, value in enumerate(data_type.table[:64]) if value != '*'
        ]
        triplets = (numpy.array(data_type.triplets)[coding_indices]).tolist()
        aa = numpy.array(list(data_type.table))[coding_indices]
        self.synonymous = torch.zeros(
            int(
                (data_type.state_count * data_type.state_count - data_type.state_count)
                / 2
            )
        )
        self.non_synonymous = torch.zeros_like(self.synonymous)
        self.transitions = torch.zeros_like(self.synonymous)
        for i, (codon1, codon2) in enumerate(combinations(triplets, 2)):
            diff = [int(a != b) for a, b in zip(codon1, codon2)]
            if sum(diff) == 1:
                index = diff.index(1)
                if codon1[index] + codon2[index] in ('AG', 'GA', 'CT', 'TC'):
                    self.transitions[i] = 1.0
                if aa[triplets.index(codon1)] == aa[triplets.index(codon2)]:
                    self.synonymous[i] = 1.0
                else:
                    self.non_synonymous[i] = 1.0

    def q(self) -> torch.Tensor:
        dim = self.data_type.state_count
        sample_shape = self.sample_shape
        ones = torch.ones(
            sample_shape + (1,), dtype=self.kappa.dtype, device=self.kappa.device
        )
        kappa, alpha, beta = torch.broadcast_tensors(
            self.kappa.tensor, self.alpha.tensor, self.beta.tensor
        )
        R = torch.zeros(sample_shape + (dim, dim), dtype=self.alpha.dtype)
        triu_indices = torch.triu_indices(row=dim, col=dim, offset=1)
        off_diagonal = (
            (torch.where(self.transitions == 1.0, kappa, ones))
            * (torch.where(self.synonymous == 1.0, alpha, ones))
            * (torch.where(self.non_synonymous == 1.0, beta, ones))
        )
        R[..., triu_indices[0], triu_indices[1]] = off_diagonal
        R[..., triu_indices[1], triu_indices[0]] = off_diagonal
        Q = R @ self.frequencies.diag_embed()
        Q[..., range(dim), range(dim)] = -torch.sum(Q, dim=-1)
        return Q

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    def handle_parameter_changed(
        self, variable: AbstractParameter, index, event
    ) -> None:
        self.fire_parameter_changed()

    @classmethod
    def from_json(cls, data, dic):
        data_type = process_object(data['data_type'], dic)
        kappa = process_object(data['kappa'], dic)
        alpha = process_object(data['alpha'], dic)
        beta = process_object(data['beta'], dic)
        frequencies = process_object(data['frequencies'], dic)
        return cls(data['id'], data_type, alpha, beta, kappa, frequencies)
