from __future__ import annotations

import math
from typing import Any

import torch

from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.utils import process_object, register_class
from torchtree.distributions.distributions import DistributionModel
from torchtree.typing import ID


@register_class
class VR(CallableModel):
    r"""Class representing the variational Renyi bound.

    VR extends traditional variational inference to Rényi’s
    :math:`\alpha`-divergences :footcite:p:`li2016renyi`.

    :param str id_: identifier of object.
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples to form estimator.
    :param float alpha: order of :math:`\alpha`-divergence.

    .. footbibliography::
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

    def _sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data: dict[str, Any], dic: dict[str, Identifiable]) -> VR:
        r"""Creates a VR object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a VR object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

        Mandatory:
          - id (str): unique string identifier.
          - variational (dict or str): variational distribution.
          - joint (dict or str): joint distribution.

        Optional:
          - samples (int or list of ints): number of samples
          - alpha (float): order of :math:`\alpha`-divergence (Default: 0).
        """
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
