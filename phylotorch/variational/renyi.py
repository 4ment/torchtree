import math

import torch

from ..core.model import CallableModel
from ..core.utils import process_object
from ..distributions.distributions import DistributionModel
from ..typing import ID


class RenyiELBO(CallableModel):

    def __init__(self, id_: ID, q: DistributionModel, p: CallableModel, samples: torch.Size, alpha: float):
        self.q = q
        self.p = p
        self.samples = samples
        self.alpha = alpha
        super(RenyiELBO, self).__init__(id_)

    def _call(self, *args, **kwargs):
        samples = kwargs.get('samples', self.samples)
        self.q.rsample(samples)
        log_w = (1. - self.alpha) * (self.p() - self.q())
        log_w_mean = torch.logsumexp(log_w, dim=-1) - math.log(log_w.shape[-1])
        return log_w_mean.sum() / (1. - self.alpha)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
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
