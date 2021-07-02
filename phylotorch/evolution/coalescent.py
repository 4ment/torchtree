import numpy
import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from .tree_model import TreeModel, TimeTreeModel
from ..core.model import CallableModel, Parameter
from ..core.utils import process_object
from ..typing import ID


class ConstantCoalescentModel(CallableModel):

    def __init__(self, id_: ID, theta: Parameter, node_heights: Parameter, sampling_times: torch.Tensor) -> None:
        super(ConstantCoalescentModel, self).__init__(id_)
        self.theta = theta
        self.sampling_times = sampling_times
        self.node_heights = node_heights
        self.add_parameter(theta)
        self.add_parameter(node_heights)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def _call(self) -> torch.Tensor:
        coalescent = ConstantCoalescent(self.sampling_times, self.theta.tensor)
        return coalescent.log_prob(self.node_heights.tensor)

    @property
    def batch_shape(self):
        return self.theta.tensor.shape[-1:]

    @property
    def sample_shape(self):
        return self.theta.tensor.shape[:-len(self.batch_shape)]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        theta = process_object(data['theta'], dic)
        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            sampling_times = tree_model.sampling_times
            node_heights = tree_model._node_heights
        else:
            times = numpy.array(data['times'])
            events = numpy.array(data['events'])
            sampling_times = torch.tensor(times[events == 1], dtype=theta.dtype)
            node_heights = torch.tensor(times[events == 0], dtype=theta.dtype)
        return cls(id_, theta, node_heights, sampling_times)


class ConstantCoalescent(Distribution):
    arg_constraints = {'theta': constraints.positive,
                       'sampling_times': constraints.greater_than_eq(0.0)}
    support = constraints.positive
    has_rsample = True

    def __init__(self, sampling_times: torch.Tensor, theta: torch.Tensor, validate_args=None) -> None:
        self.sampling_times = sampling_times
        self.theta = theta
        self.taxon_count = sampling_times.shape
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super(ConstantCoalescent, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, sampling_times: torch.Tensor, thetas: torch.Tensor, validate_args=None) -> None:
        super(PiecewiseConstantCoalescent, self).__init__(sampling_times, thetas, validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
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
        # TODO: why was this so complicated?
        # log_thetas = torch.where(node_mask_sorted[..., 1:] == -1, torch.log(thetas),
        #                          torch.zeros(1, dtype=heights.dtype))
        # return -torch.sum(lchoose2 * durations / thetas, -1, keepdim=True) - log_thetas.sum(-1, keepdim=True)
        return -torch.sum(lchoose2 * durations / thetas, -1, keepdim=True) - self.theta.log().sum(-1, keepdim=True)


class PiecewiseConstantCoalescentModel(ConstantCoalescentModel):
    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescent(self.sampling_times, self.theta.tensor)
        return pwc.log_prob(self.node_heights.tensor)


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(self, sampling_times: torch.Tensor, thetas: torch.Tensor, grid: torch.Tensor,
                 validate_args=None) -> None:
        super(PiecewiseConstantCoalescentGrid, self).__init__(sampling_times, thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        if node_heights.dim() > self.theta.dim():
            batch_shape = node_heights.shape[:-1]
        else:
            batch_shape = self.theta.shape[:-1]

        sampling_times = self.sampling_times.expand(batch_shape + torch.Size([-1]))
        grid = self.grid.expand(batch_shape + torch.Size([-1]))

        if node_heights.dim() < self.theta.dim():
            heights = torch.cat([sampling_times, node_heights.expand(batch_shape + torch.Size([-1])), grid], -1)
        else:
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
    def __init__(self, id_: ID, theta: Parameter, node_heights: Parameter, sampling_times: torch.Tensor,
                 grid: Parameter) -> None:
        self.grid = grid
        super(PiecewiseConstantCoalescentGridModel, self).__init__(id_, theta, node_heights, sampling_times)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescentGrid(self.sampling_times, self.theta.tensor, self.grid.tensor)
        return pwc.log_prob(self.node_heights.tensor)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']

        theta = process_object(data['theta'], dic)

        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            sampling_times = tree_model.sampling_times
            node_heights = tree_model._node_heights
        else:
            times = numpy.array(data['times'])
            events = numpy.array(data['events'])
            sampling_times = torch.tensor(times[events == 1], dtype=theta.dtype)
            node_heights = torch.tensor(times[events == 0], dtype=theta.dtype)

        if 'grid' not in data:
            cutoff = data['cutoff']  # float
            grid = Parameter(None, torch.tensor(np.linspace(0, cutoff, num=theta.shape[0])[1:]))
        else:
            if isinstance(data['grid'], list):
                grid = Parameter(None, torch.tensor(data['grid'], dtype=theta.dtype))
            else:
                grid = process_object(data['grid'], dic)
            assert grid.shape[0] + 1 == theta.shape[0]

        return cls(id_, theta, node_heights, sampling_times, grid)
