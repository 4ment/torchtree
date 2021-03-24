import torch

from ..core.model import CallableModel
from ..core.utils import process_object


# Code adapted from https://github.com/beast-dev/beast-mcmc/blob/master/src/dr/evomodel/tree/CTMCScalePrior.java
class CTMCScale(CallableModel):
    shape = torch.tensor([0.5])
    log_gamma_one_half = torch.lgamma(shape)

    def __init__(self, id_, x, tree_model):
        super(CTMCScale, self).__init__(id_)
        self.x = x
        self.tree_model = tree_model
        self.add_parameter(x)

    def __call__(self, *args, **kwargs):
        total_tree_time = self.tree_model.branch_lengths().sum()
        log_normalization = self.shape * torch.log(total_tree_time) - self.log_gamma_one_half
        log_like = log_normalization - self.shape * self.x.tensor.log() - self.x.tensor * total_tree_time
        return log_like.sum()

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_object(data['x'], dic)
        tree_model = process_object(data['tree_model'], dic)
        return cls(id_, x, tree_model)
