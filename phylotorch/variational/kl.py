import torch

from ..core.utils import process_object
from ..core.model import CallableModel


class ELBO(CallableModel):

    def __init__(self, id_, q, p, samples, forward=False):
        self.q = q
        self.p = p
        self.samples = samples
        self.forward = forward
        super(ELBO, self).__init__(id_)

    def _call(self, *args, **kwargs):
        samples = kwargs.get('samples', self.samples)
        if self.forward:
            if isinstance(samples, torch.Size):
                self.q.rsample(samples)
                log_w = self.p() - self.q()
                log_w_norm = log_w - torch.logsumexp(log_w, -1)
                lp = torch.sum(log_w_norm.exp() * log_w)
            else:
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
                lp = torch.sum(w_norm * log_w)
        else:
            if isinstance(samples, torch.Size):
                self.q.rsample(samples)
                lp = (self.p() - self.q()).mean()
            else:
                elbos = []
                for i in range(samples):
                    self.q.rsample()
                    elbos.append(self.p() - self.q())
                lp = torch.stack(elbos).mean()
        return lp

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

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        forward_kl = data.get('forward', False)

        return cls(data['id'], var, joint, samples, forward_kl)
