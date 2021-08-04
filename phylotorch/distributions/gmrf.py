import torch
import torch.distributions.normal

from ..core.model import CallableModel, Parameter
from ..core.utils import process_object
from ..evolution.tree_model import TimeTreeModel, TreeModel
from ..typing import ID


class GMRF(CallableModel):
    def __init__(
        self,
        id_: ID,
        x: Parameter,
        precision: Parameter,
        tree_model: TimeTreeModel = None,
        weights: torch.Tensor = None,
        rescale: bool = True,
    ) -> None:
        super(GMRF, self).__init__(id_)
        self.tree_model = tree_model
        self.weights = weights
        self.x = x
        self.precision = precision
        self.add_parameter(self.x)
        self.rescale = rescale

    def _call(self, *args, **kwargs):
        s = torch.pow(self.x.tensor[..., :-1] - self.x.tensor[..., 1:], 2.0)
        if self.tree_model is not None:
            heights = torch.cat(
                (
                    torch.zeros(self.tree_model.node_heights.shape[:-1] + (1,)),
                    self.tree_model.node_heights,
                ),
                -1,
            )
            indices = torch.argsort(heights, descending=False)
            heights_sorted = torch.gather(heights, -1, indices)
            durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
            s /= (durations[..., :-1] + durations[..., 1:]) / 2.0
            if self.rescale:
                s *= heights_sorted[..., -1:]
        elif self.weights is not None:
            s /= self.weights

        dim = self.x.shape[-1] - 1.0  # field dim
        precision = self.precision.tensor
        return (
            precision.log() * dim / 2.0
            - s.sum(-1, keepdim=True) * precision / 2.0
            - dim / 2.0 * 1.8378770664093453
        )

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
        tree_model = None
        weights = None
        # time-aware if a tree_model is provided
        if TreeModel.tag in data:
            tree_model = process_object(data[TreeModel.tag], dic)
        elif 'weights' in data:
            weights = process_object(data['weights'], dic)
        x = process_object(data['x'], dic)
        precision = process_object(data['precision'], dic)
        rescale = data.get('rescale', True)
        return cls(id_, x, precision, tree_model, weights, rescale)
