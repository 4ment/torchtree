import torch

from ..core.model import CallableModel
from phylotorch.core.utils import process_object


class CUBO(CallableModel):

    def __init__(self, id_, q, p, n, samples):
        self.q = q
        self.p = p
        self.n = n
        self.samples = samples
        super(CUBO, self).__init__(id_)

    def __call__(self):
        p_log_prob = []
        q_log_prob = []
        for i in range(self.samples):
            self.q.rsample()
            q_log_prob.append(self.q())
            p_log_prob.append(self.p())
        p_log_prob = torch.stack(p_log_prob)
        q_log_prob = torch.stack(q_log_prob)
        log_w = float(self.n) * (p_log_prob - q_log_prob)
        return (torch.logsumexp(log_w, 0) - torch.log(torch.tensor(float(self.samples)))) / self.n

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        samples = data.get('samples', 1)
        n = data.get('n', 2.0)

        var_desc = data['variational']
        var = process_object(var_desc, dic)

        joint_desc = data['joint']
        joint = process_object(joint_desc, dic)

        return cls(data['id'], var, joint, samples, n)
