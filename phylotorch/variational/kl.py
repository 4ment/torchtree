from ..core.model import CallableModel
from phylotorch.core.utils import process_object
import torch


class ELBO(CallableModel):

    def __init__(self, id_, q, p, samples, forward=False):
        self.q = q
        self.p = p
        self.samples = samples
        self.forward = forward
        super(ELBO, self).__init__(id_)

    def __call__(self, *args, **kwargs):
        samples = kwargs.get('samples', self.samples)
        if self.forward:
            p_log_prob = []
            q_log_prob = []
            for i in range(samples):
                self.q.rsample()
                q_log_prob.append(self.q())
                p_log_prob.append(self.p())
            p_log_prob = torch.stack(p_log_prob)
            q_log_prob = torch.stack(q_log_prob)
            log_w = p_log_prob - q_log_prob
            log_w_norm = log_w - torch.logsumexp(log_w, 0)
            w_norm = log_w_norm.exp()
            return torch.sum(w_norm * log_w)
        else:
            elbos = []
            for i in range(self.samples):
                self.q.rsample()
                elbos.append(self.p() - self.q())
        return torch.stack(elbos).sum()/self.samples

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        samples = data.get('samples', 1)

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        return cls(data['id'], var, joint, samples)