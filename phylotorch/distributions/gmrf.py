import torch
import torch.distributions.normal

from ..core.model import CallableModel
from ..core.utils import process_object
from ..evolution.tree_model import TreeModel


class GMRF(CallableModel):
    def __init__(self, id_, x, precision, tree_model):
        super(GMRF, self).__init__(id_)
        self.tree_model = tree_model
        self.x = x
        self.precision = precision
        self.add_parameter(self.x)

    def __call__(self, *args, **kwargs):
        return torch.distributions.normal.Normal(0.0, torch.tensor([1.0]) / self.precision.tensor.sqrt()).log_prob(
            self.x.tensor[:-1] - self.x.tensor[1:]).sum()

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        x = process_object(data['x'], dic)
        precision = process_object(data['precision'], dic)
        return cls(id_, x, precision, tree_model)
