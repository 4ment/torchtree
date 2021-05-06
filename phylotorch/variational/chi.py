import torch

from ..core.model import CallableModel
from ..core.utils import process_object
from ..distributions.distributions import DistributionModel
from ..typing import ID


class CUBO(CallableModel):

    def __init__(self, id_: ID, q: DistributionModel, p: CallableModel, samples: torch.Size, n: torch.Tensor):
        self.q = q
        self.p = p
        self.n = n
        self.samples = samples
        super(CUBO, self).__init__(id_)

    def _call(self, *args, **kwargs):
        samples = kwargs.get('samples', self.samples)
        self.q.rsample(samples)
        log_w = self.p() - self.q()
        log_max = torch.max(log_w)
        log_w_rescaled = torch.exp(log_w - log_max) ** self.n
        return torch.log(log_w_rescaled.mean()) / self.n + log_max

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
        n = torch.tensor(data.get('n', 2.0))

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        return cls(data['id'], var, joint, samples, n)
