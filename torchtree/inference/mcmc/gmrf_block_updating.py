from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from torchtree.core.identifiable import Identifiable
from torchtree.core.utils import process_objects, register_class
from torchtree.distributions.gmrf import GMRF
from torchtree.evolution.coalescent import AbstractCoalescentModel
from torchtree.inference.mcmc.operator import MCMCOperator
from torchtree.typing import ID

# Code adapted from
# github.com/beast-dev/beast-mcmc/blob/master/src/dr/evomodel/coalescent/operators/GMRFSkygridBlockUpdateOperator.java


@register_class
class GMRFPiecewiseCoalescentBlockUpdatingOperator(MCMCOperator):
    r"""Class implementing the block-updating Markov chain Monte Carlo sampling
    for Gaussian Markov random fields (GMRF).

    :param ID id_: identifier of object.
    :param coalescent: coalescent object.
    :param GMRF gmrf: GMRF object.
    :param float weight: operator weight.
    :param float target_acceptance_probability: target acceptance probability.
    :param float scaler: scaler for tuning the precision parameter proposal.
    """

    def __init__(
        self,
        id_: ID,
        coalescent: AbstractCoalescentModel,
        gmrf: GMRF,
        weight: float,
        target_acceptance_probability: float,
        scaler: float,
        **kwargs,
    ):
        super().__init__(
            id_,
            [gmrf.field, gmrf.precision],
            weight,
            target_acceptance_probability,
            **kwargs,
        )
        self.gmrf = gmrf
        self.coalescent = coalescent
        self._stop_value = kwargs.get("stop_value", 0.1)
        self._max_iterations = kwargs.get("max_iterations", 200)
        self._scaler = scaler

    @property
    def tuning_parameter(self) -> float:
        return self._scaler

    @MCMCOperator.adaptable_parameter.getter
    def adaptable_parameter(self) -> float:
        return math.sqrt(self._scaler - 1)

    def set_adaptable_parameter(self, value: float) -> None:
        self._scaler = 1 + value * value

    def propose_precision(self):
        length = self._scaler - 1 / self._scaler
        if self._scaler == 1:
            new_precision = self.gmrf.precision.tensor
        elif torch.rand(1) < length / (length + 2 * math.log(self._scaler)):
            new_precision = (
                1 / self._scaler + length * torch.rand(1)
            ) * self.gmrf.precision.tensor
        else:
            new_precision = (
                math.pow(self._scaler, 2.0 * torch.rand(1) - 1)
                * self.gmrf.precision.tensor
            )
        return new_precision

    def jacobian(self, wNative, gamma, precision_matrix):
        jacobian = precision_matrix.clone()
        dim = jacobian.shape[-1]
        jacobian[range(dim), range(dim)] += torch.exp(-gamma) * wNative
        return jacobian

    def gradient(self, numCoalEv, wNative, gamma, precision_matrix):
        return -(precision_matrix @ gamma) - numCoalEv + torch.exp(-gamma) * wNative

    def newton_raphson(self, numCoalEv, wNative, gamma, precision_matrix):
        iteration = 0
        grad = torch.tensor(float("inf"))
        gamma = gamma.clone()
        while (
            torch.linalg.vector_norm(grad) > self._stop_value
            and iteration < self._max_iterations
        ):
            jac = self.jacobian(wNative, gamma, precision_matrix)
            grad = self.gradient(numCoalEv, wNative, gamma, precision_matrix)
            gamma = gamma + torch.linalg.solve(jac, grad)
            iteration += 1
        return gamma

    def __call__(self) -> Tensor:
        coalescent = self.coalescent.distribution()
        gamma = self.gmrf.field.tensor
        sufficient_statistics, coalescent_counts = coalescent.sufficient_statistics(
            self.coalescent.tree_model.node_heights
        )
        self.gmrf.precision.tensor = self.propose_precision()
        proposed_precision_matrix = self.gmrf.precision_matrix()

        if gamma.dim() > 1:
            mode_forward = []
            for i in range(gamma.shape[-2]):
                mode = self.newton_raphson(
                    coalescent_counts[i],
                    sufficient_statistics[i],
                    gamma[i],
                    proposed_precision_matrix[i],
                )
                mode_forward.append(mode)
            mode_forward = torch.stack(mode_forward)
        else:
            mode_forward = self.newton_raphson(
                coalescent_counts,
                sufficient_statistics,
                gamma,
                proposed_precision_matrix,
            )

        forwardQW = proposed_precision_matrix.clone()

        dim = gamma.shape[-1]

        diagonal1 = sufficient_statistics * torch.exp(-mode_forward)
        forwardQW[..., range(dim), range(dim)] += diagonal1
        diagonal1 = diagonal1 * (mode_forward + 1) - coalescent_counts

        z = torch.randn(gamma.shape)

        cholesky = torch.linalg.cholesky(forwardQW, upper=True)

        v = torch.linalg.solve(cholesky.transpose(-1, -2), diagonal1)
        mu = torch.linalg.solve(cholesky, v)
        u = torch.linalg.solve(cholesky, z)
        proposed_gamma = mu + u

        self.gmrf.field.tensor = proposed_gamma.clone()

        diagonal = cholesky[..., range(dim), range(dim)]
        diagonal = torch.where(diagonal > 0.0000001, diagonal.log(), torch.zeros([1]))

        log_q_forward = diagonal.sum(-1) - 0.5 * (z * z).sum(-1)
        return log_q_forward

    def _step(self) -> Tensor:
        coalescent = self.coalescent.distribution()
        gamma = self.gmrf.field.tensor
        sufficient_statistics, coalescent_counts = coalescent.sufficient_statistics(
            self.coalescent.tree_model.node_heights
        )
        precision_matrix = self.gmrf.precision_matrix()
        self.gmrf.precision.tensor = self.propose_precision()
        proposed_precision_matrix = self.gmrf.precision_matrix()

        mode_forward = self.newton_raphson(
            coalescent_counts, sufficient_statistics, gamma, proposed_precision_matrix
        )

        forwardQW = proposed_precision_matrix.clone()
        backwardQW = precision_matrix.clone()

        dim = precision_matrix.shape[-1]

        diagonal1 = sufficient_statistics * torch.exp(-mode_forward)
        forwardQW[range(dim), range(dim)] += diagonal1
        diagonal1 = diagonal1 * (mode_forward + 1) - coalescent_counts

        z = torch.randn(dim)

        try:
            cholesky = torch.linalg.cholesky(forwardQW, upper=True)
        except torch._C._LinAlgError:
            return torch.tensor(torch.inf)

        v = torch.linalg.solve(cholesky.t(), diagonal1)
        mu = torch.linalg.solve(cholesky, v)
        u = torch.linalg.solve(cholesky, z)
        proposed_gamma = mu + u

        self.gmrf.field.tensor = proposed_gamma.clone()

        diagonal = cholesky[range(dim), range(dim)]
        log_q_forward = diagonal[diagonal > 0.0000001].log().sum() - 0.5 * (z @ z)

        mode_backward = self.newton_raphson(
            coalescent_counts, sufficient_statistics, proposed_gamma, precision_matrix
        )

        diagonal1 = sufficient_statistics * torch.exp(-mode_backward)
        backwardQW[range(dim), range(dim)] += diagonal1
        diagonal1 = diagonal1 * (mode_backward + 1) - coalescent_counts

        try:
            cholesky = torch.linalg.cholesky(backwardQW, upper=True)
        except torch._C._LinAlgError:
            return torch.tensor(torch.inf)

        v = torch.linalg.solve(cholesky.t(), diagonal1)
        mu = torch.linalg.solve(cholesky, v)
        diagonal1 = gamma - mu
        diagonal3 = backwardQW @ diagonal1

        diagonal = cholesky[range(dim), range(dim)]
        log_q_backward = diagonal[diagonal > 0.0000001].log().sum() - 0.5 * (
            diagonal1 @ diagonal3
        )
        return log_q_backward - log_q_forward

    @classmethod
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> GMRFPiecewiseCoalescentBlockUpdatingOperator:
        r"""Creates a GMRFPiecewiseCoalescentBlockUpdatingOperator object from a
        dictionary.

        :param dict[str, Any] data: dictionary representation of a GMRFCovariate
            object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - coalescent (dict or str): coalescent model.
          - gmrf (dict or str): GMRF model.

         Optional:
          - weight (float): weight of operator (Default: 1)
          - scaler (float): rescale by root height (Default: 2.0).
          - target_acceptance_probability (float): target acceptance
            probability (Default: 0.24).
          - disable_adaptation (bool): disable adaptation (Default: false).

        .. note::
            The precision proposal is not tuned if the scaler is equal to 1.
        """
        id_ = data["id"]
        coalescent = process_objects(data["coalescent"], dic)
        gmrf = process_objects(data["gmrf"], dic)
        weight = data.get("weight", 1.0)
        scaler = data.get("scaler", 2.0)
        target_acceptance_probability = data.get("target_acceptance_probability", 0.24)
        optionals = {}
        optionals["disable_adaptation"] = data.get("disable_adaptation", False)

        return cls(
            id_,
            coalescent,
            gmrf,
            weight,
            target_acceptance_probability,
            scaler,
            **optionals,
        )
