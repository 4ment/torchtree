r"""Reversible nucleotide substitution models.

Reversible nucleotide substitution models are characterized by the following:

1. **Time Reversibility**

The substitution process satisfies the detailed balance condition:

.. math::

    \pi_i Q_{ij} = \pi_j Q_{ji}

where:

- :math:`\pi_i` and :math:`\pi_j` are the equilibrium frequencies of nucleotides :math:`i` and :math:`j`.
- :math:`Q_{ij}` is the rate of substitution from nucleotide :math:`i` to :math:`j`.

This ensures that the process appears the same forward and backward in time, making the likelihood computations simpler.

2. **Equilibrium Frequencies**

Represent the long-term stationary distribution of nucleotides. These frequencies account for biases in nucleotide composition.

3. **Rate Matrix (Q)**

The general structure of the rate matrix for reversible models is:

.. math::

    Q_{ij} =
    \begin{cases}
    \mu_{ij} \pi_j & \text{if } i \neq j \\
    -\sum_{k \neq i} Q_{ik} & \text{if } i = j
    \end{cases}

where:

- :math:`\mu_{ij}` is the relative rate of substitution between nucleotides :math:`i` and :math:`j`.
- Diagonal entries (:math:`Q_{ii}`) ensure rows sum to zero.

4. **Scaling**

The rate matrix is typically scaled such that the average rate of substitution is 1.
The scaling factor :math:`\beta` is given by:

.. math::
    \beta = -\frac{1}{\sum_{i} \pi_i \mu_{ii}}

.. note::
    The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be :math:`\pi_A, \pi_C, \pi_G, \pi_T`.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.linalg
from torch import Tensor

from ...core.abstractparameter import AbstractParameter
from ...core.utils import process_object, register_class
from ...typing import ID
from .abstract import SubstitutionModel, SymmetricSubstitutionModel


@register_class
class JC69(SubstitutionModel):
    r"""Jukes-Cantor (JC69) substitution model.
    
    The JC69 model assumes:
    
    - **Equal substitution rates:** Each nucleotide is equally likely to mutate into another nucleotide.
    - **Equal base frequencies:** The equilibrium frequencies of :math:`\pi_A, \pi_C, \pi_G, \pi_T` are all equal to 0.25.
    - **Reversibility:** The substitution process is time-reversible.
    
    The JC69 rate matrix :math:`Q` is given as:

    .. math::

        Q = 
        \begin{bmatrix}
        -1 & 1/3 & 1/3 & 1/3 \\
        1/3 & -1 & 1/3 & 1/3 \\
        1/3 & 1/3 & -1 & 1/3 \\
        1/3 & 1/3 & 1/3 & -1
        \end{bmatrix}

    """

    def __init__(self, id_: ID) -> None:
        super().__init__(id_)
        self._frequencies = torch.full((4,), 0.25)

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies

    @property
    def rates(self) -> Union[Tensor, list[Tensor]]:
        return []

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

    def _sample_shape(self) -> torch.Size:
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
    r"""Hasegawa-Kishino-Yano (HKY) substitution model.
    
    The HKY model has:

    - A transition/transversion rate ratio parameters: :math:`\kappa`.
    - Four equilibrium frequency parameters: :math:`\pi_A, \pi_C, \pi_G, \pi_T`.
    
    The HKY rate matrix :math:`Q` is given as:

    .. math::

        Q = 
        \begin{bmatrix}
        -(\pi_C + \kappa \pi_G + \pi_T) & \pi_C & \kappa \pi_G & \pi_T \\
        \pi_A & -(\pi_A + \pi_G + \kappa \pi_T) & \pi_G & \kappa \pi_T \\
        \kappa \pi_A & \pi_C & -(\kappa \pi_A + \pi_C + \pi_T) & \pi_T \\
        \pi_A & \kappa \pi_C & \pi_G & -(\pi_A + \kappa \pi_C + \pi_G)
        \end{bmatrix}
    """

    def __init__(
        self, id_: ID, kappa: AbstractParameter, frequencies: AbstractParameter
    ) -> None:
        super().__init__(id_, frequencies)
        self._kappa = kappa

    @property
    def rates(self) -> Union[Tensor, list[Tensor]]:
        return self._kappa.tensor

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
            pi = self.frequencies.expand(self.kappa.shape[:-1] + (4,)).unsqueeze(-2)
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
    r"""General Time Reversible (GTR) substitution model.
    
    The GTR model has:

    - Six substitution rate parameters: :math:`a, b, c, d, e, f`.
    - Four equilibrium frequency parameters: :math:`\pi_A, \pi_C, \pi_G, \pi_T`.
    
    The GTR rate matrix :math:`Q` is given as:

    .. math::

        Q = 
        \begin{bmatrix}
        -(a \pi_C + b \pi_G + c \pi_T) & a \pi_C & b \pi_G & c \pi_T \\
        a \pi_A & -(a \pi_A + d \pi_G + e \pi_T) & d \pi_G & e \pi_T \\
        b \pi_A & d \pi_C & -(b \pi_A + d \pi_C + f \pi_T) & f \pi_T \\
        c \pi_A & e \pi_C & f \pi_G & -(c \pi_A + e \pi_C + f \pi_G)
        \end{bmatrix}

    where the exchangeability parameters are defined as:

    .. math::
        \begin{align*}
        a &= r_{AC} = r_{CA}\\
        b &= r_{AG} = r_{GA}\\
        c &= r_{AT} = r_{TA}\\
        d &= r_{CG} = r_{GC}\\
        e &= r_{CT} = r_{TC}\\
        f &= r_{GT} = r_{TG}
        \end{align*}
    
    .. note::
        The order of the rate parameters in a :class:`~torchtree.Parameter` is expected to be :math:`a, b, c, d, e, f`.
        The upper off-diagonal elements are indexed by first iterating over rows (row 0, then row 1, etc.) and then over columns for each row.
    """

    def __init__(
        self, id_: ID, rates: AbstractParameter, frequencies: AbstractParameter
    ):
        super().__init__(id_, frequencies)
        self._rates = rates

    @property
    def rates(self) -> Union[Tensor, list[Tensor]]:
        return self._rates.tensor

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def q(self) -> torch.Tensor:
        if len(self.frequencies.shape[:-1]) != len(self.rates.shape[:-1]):
            pi = self.frequencies.unsqueeze(0).unsqueeze(-2)
            rates = self.rates.unsqueeze(-2)
        elif len(self.frequencies.shape) == 1:
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
