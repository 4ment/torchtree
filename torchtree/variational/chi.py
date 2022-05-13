from __future__ import annotations

import torch

from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..distributions.distributions import DistributionModel
from ..typing import ID


@register_class
class CUBO(CallableModel):
    r"""
    Class representing the :math:`\chi`-upper bound (CUBO) objective [#Dieng2017]_.

    :param id_: unique identifier of object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples to form estimator.
    :param torch.Tensor n: order of :math:`\chi`-divergence.

    .. [#Dieng2017] Adji Bousso Dieng, Dustin Tran, Rajesh Ranganath, John Paisley,
     David Blei. Variational Inference
     via :math:`\chi` Upper Bound Minimization
    """

    def __init__(
        self,
        id_: ID,
        q: DistributionModel,
        p: CallableModel,
        samples: torch.Size,
        n: torch.Tensor,
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.n = n
        self.samples = samples

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        self.q.rsample(samples)
        log_w = self.p() - self.q()
        log_max = torch.max(log_w)
        log_w_rescaled = torch.exp(log_w - log_max) ** self.n
        return torch.log(log_w_rescaled.mean()) / self.n + log_max

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> CUBO:
        samples = data.get('samples', 1)
        if isinstance(samples, list):
            samples = torch.Size(samples)
        else:
            samples = torch.Size((samples,))
        n = torch.tensor(data.get('n', 2.0))

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        return cls(data['id'], var, joint, samples, n)
