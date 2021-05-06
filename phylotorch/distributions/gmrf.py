import torch
import torch.distributions.normal

from ..core.model import CallableModel, Parameter
from ..core.utils import process_object
from ..evolution.tree_model import TreeModel
from ..typing import ID


class GMRF(CallableModel):
    def __init__(self, id_: ID, x: Parameter, precision: Parameter, tree_model: TreeModel = None) -> None:
        super(GMRF, self).__init__(id_)
        self.tree_model = tree_model
        self.x = x
        self.precision = precision
        self.add_parameter(self.x)

    def _call(self, *args, **kwargs):
        return torch.distributions.normal.Normal(0.0, torch.tensor([1.0]) / self.precision.tensor.sqrt()).log_prob(
            self.x.tensor[..., :-1] - self.x.tensor[..., 1:]).sum(-1, keepdim=True)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @property
    def batch_shape(self):
        return self.x.tensor.shape[-1:]

    @property
    def sample_shape(self):
        return self.x.tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        if TreeModel.tag in data:
            tree_model = process_object(data[TreeModel.tag], dic)
        else:
            tree_model = None
        x = process_object(data['x'], dic)
        precision = process_object(data['precision'], dic)
        return cls(id_, x, precision, tree_model)
