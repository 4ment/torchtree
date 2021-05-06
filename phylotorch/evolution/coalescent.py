import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from .tree_model import TreeModel
from ..core.model import CallableModel, Parameter
from ..core.utils import process_object


class ConstantCoalescentModel(CallableModel):

    def __init__(self, id_, theta, tree):
        super(ConstantCoalescentModel, self).__init__(id_)
        self.theta = theta
        self.tree = tree
        self.add_parameter(theta)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def _call(self):
        coalescent = ConstantCoalescent(self.tree.sampling_times, self.theta.tensor)
        return coalescent.log_prob(self.tree.node_heights)

    @property
    def batch_shape(self):
        return self.theta.tensor.shape[-1:]

    @property
    def sample_shape(self):
        return self.theta.tensor.shape[:-len(self.batch_shape)]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        theta = process_object(data['theta'], dic)
        return cls(id_, theta, tree_model)


class ConstantCoalescent(Distribution):
    arg_constraints = {'theta': constraints.positive,
                       'sampling_times': constraints.greater_than_eq(0.0)}
    support = constraints.positive
    has_rsample = True

    def __init__(self, sampling_times, theta, validate_args=None):
        self.sampling_times = sampling_times
        self.theta = theta
        self.taxon_count = sampling_times.shape
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super(ConstantCoalescent, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights):
        sampling_times = self.sampling_times if node_heights.dim() == 1 else self.sampling_times.expand(
            node_heights.shape[:-1] + torch.Size([-1]))
        heights = torch.cat([sampling_times, node_heights], dim=-1)
        node_mask = torch.cat([torch.full(sampling_times.shape, False, dtype=torch.bool),
                               torch.full(sampling_times.shape[:-1] + torch.Size([sampling_times.shape[-1] - 1]), True,
                                          dtype=torch.bool)],
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = torch.where(node_mask_sorted, torch.full_like(self.theta, -1),
                                    torch.full_like(self.theta, 1)).cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(-lchoose2 * durations / self.theta, -1, keepdim=True) - (self.taxon_count[-1] - 1) * torch.log(
            self.theta)

    def rsample(self, sample_shape=torch.Size()):
        lineage_count = torch.arange(self.taxon_count[-1], 1, -1, dtype=self.theta.dtype)
        return torch.distributions.Exponential(lineage_count * (lineage_count - 1) / (2.0 * self.theta)).rsample(
            sample_shape)


class PiecewiseConstantCoalescent(ConstantCoalescent):
    def __init__(self, sampling_times, thetas, validate_args=None):
        super(PiecewiseConstantCoalescent, self).__init__(sampling_times, thetas, validate_args)

    def log_prob(self, node_heights):
        sampling_times = self.sampling_times if node_heights.dim() == 1 else self.sampling_times.expand(
            node_heights.shape[:-1] + torch.Size([-1]))

        heights = torch.cat([sampling_times, node_heights], -1)
        node_mask = torch.cat([torch.full(sampling_times.shape, 1, dtype=torch.int),  # sampling event
                               torch.full(sampling_times.shape[:-1] + torch.Size([sampling_times.shape[-1] - 1]), -1,
                                          dtype=torch.int)],  # coalescent event
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(node_mask_sorted == -1, torch.tensor([1], dtype=torch.long),
                                     torch.tensor([0], dtype=torch.long)).cumsum(-1)[..., :-1]

        thetas = self.theta.gather(-1, thetas_indices)
        log_thetas = torch.where(node_mask_sorted[..., 1:] == -1, torch.log(thetas),
                                 torch.zeros(1, dtype=heights.dtype))
        return -torch.sum(lchoose2 * durations / thetas, -1, keepdim=True) - log_thetas.sum(-1, keepdim=True)


class PiecewiseConstantCoalescentModel(ConstantCoalescentModel):
    def _call(self, *args, **kwargs):
        pwc = PiecewiseConstantCoalescent(self.tree.sampling_times, self.theta.tensor)
        return pwc.log_prob(self.tree.node_heights)


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(self, sampling_times, thetas, grid, validate_args=None):
        super(PiecewiseConstantCoalescentGrid, self).__init__(sampling_times, thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights):
        sampling_times = self.sampling_times if node_heights.dim() == 1 else self.sampling_times.expand(
            node_heights.shape[:-1] + torch.Size([-1]))

        grid = self.grid if node_heights.dim() == 1 else self.grid.expand(node_heights.shape[:-1] + torch.Size([-1]))

        heights = torch.cat([sampling_times, node_heights, grid], -1)
        node_mask = torch.cat([torch.full(sampling_times.shape, 1, dtype=torch.int),  # sampling event
                               torch.full(sampling_times.shape[:-1] + torch.Size([sampling_times.shape[-1] - 1]), -1,
                                          dtype=torch.int),
                               # coalescent event
                               torch.full(grid.shape, 0, dtype=torch.int)],  # no event
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(node_mask_sorted == 0, torch.tensor([1], dtype=torch.long),
                                     torch.tensor([0], dtype=torch.long)).cumsum(-1)

        thetas = self.theta.gather(-1, thetas_indices)
        log_thetas = torch.where(node_mask_sorted == -1, torch.log(thetas), torch.zeros(1, dtype=heights.dtype))
        return -torch.sum(lchoose2 * durations / thetas[..., :-1], -1, keepdim=True) - log_thetas.sum(-1, keepdim=True)


class PiecewiseConstantCoalescentGridModel(ConstantCoalescentModel):
    def __init__(self, id_, theta, tree, grid):
        self.grid = grid
        super(PiecewiseConstantCoalescentGridModel, self).__init__(id_, theta, tree)

    def _call(self, *args, **kwargs):
        pwc = PiecewiseConstantCoalescentGrid(self.tree.sampling_times, self.theta.tensor, self.grid.tensor)
        return pwc.log_prob(self.tree.node_heights)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        theta = process_object(data['theta'], dic)
        if 'grid' not in data:
            cutoff = data['cutoff']  # float
            grid = Parameter(None, torch.tensor(np.linspace(0, cutoff, num=theta.shape[0])[1:]))
        else:
            grid = process_object(data['grid'], dic)
            assert grid.shape[0] + 1 == theta.shape[0]
        return cls(id_, theta, tree_model, grid)
