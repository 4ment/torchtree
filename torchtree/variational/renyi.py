from __future__ import annotations

import math

import torch

from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..distributions.distributions import DistributionModel
from ..typing import ID


@register_class
class VR(CallableModel):
    r"""
    Class representing the variational Renyi bound (VR) [#Li2016]_.
    VR extends traditional variational inference to Rényi’s :math:`\alpha`-divergences.

    :param id_: unique identifier of object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples to form estimator.
    :param float alpha: order of :math:`\alpha`-divergence.

    .. [#Li2016] Yingzhen Li, Richard E. Turner. Rényi Divergence Variational Inference.
    """

    def __init__(
        self,
        id_: ID,
        q: DistributionModel,
        p: CallableModel,
        samples: torch.Size,
        alpha: float,
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples
        self.alpha = alpha

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        self.q.rsample(samples)
        log_w = (1.0 - self.alpha) * (self.p() - self.q())
        log_w_mean = torch.logsumexp(log_w, dim=-1) - math.log(log_w.shape[-1])
        return log_w_mean.sum(-1) / (1.0 - self.alpha)

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> VR:
        samples = data.get('samples', 1)
        if isinstance(samples, list):
            samples = torch.Size(samples)
        else:
            samples = torch.Size((samples,))

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        alpha = data.get('alpha', 0.0)

        return cls(data['id'], var, joint, samples, alpha)
