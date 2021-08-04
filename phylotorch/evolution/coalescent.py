from typing import List, Tuple, Union

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from ..core.model import CallableModel, Parameter
from ..core.utils import process_object, process_objects
from ..typing import ID
from .tree_model import TimeTreeModel, TreeModel


class ConstantCoalescentModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        theta: Parameter,
        node_heights: Parameter,
        sampling_times: torch.Tensor,
    ) -> None:
        super().__init__(id_)
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

    def _call(self, *args, **kwargs) -> torch.Tensor:
        coalescent = ConstantCoalescent(self.sampling_times, self.theta.tensor)
        return coalescent.log_prob(self.node_heights.tensor)

    @property
    def batch_shape(self):
        return self.theta.tensor.shape[-1:]

    @property
    def sample_shape(self):
        return self.theta.tensor.shape[: -len(self.batch_shape)]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        theta = process_object(data['theta'], dic)
        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            sampling_times = tree_model.sampling_times
            node_heights = tree_model._node_heights
        else:
            sampling_times, node_heights = process_times_events(
                data['times'], data['events'], theta.dtype
            )
        return cls(id_, theta, node_heights, sampling_times)


class ConstantCoalescent(Distribution):

    arg_constraints = {
        'theta': constraints.positive,
        'sampling_times': constraints.greater_than_eq(0.0),
    }
    support = constraints.positive
    has_rsample = True

    def __init__(
        self, sampling_times: torch.Tensor, theta: torch.Tensor, validate_args=None
    ) -> None:
        self.sampling_times = sampling_times
        self.theta = theta
        self.taxon_count = sampling_times.shape
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        sampling_times = (
            self.sampling_times
            if node_heights.dim() == 1
            else self.sampling_times.expand(node_heights.shape[:-1] + torch.Size([-1]))
        )
        heights = torch.cat([sampling_times, node_heights], dim=-1)
        node_mask = torch.cat(
            [
                torch.full(sampling_times.shape, False, dtype=torch.bool),
                torch.full(
                    sampling_times.shape[:-1]
                    + torch.Size([sampling_times.shape[-1] - 1]),
                    True,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = torch.where(
            node_mask_sorted,
            torch.full_like(self.theta, -1),
            torch.full_like(self.theta, 1),
        ).cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(-lchoose2 * durations / self.theta, -1, keepdim=True) - (
            self.taxon_count[-1] - 1
        ) * torch.log(self.theta)

    def rsample(self, sample_shape=torch.Size()):
        lineage_count = torch.arange(
            self.taxon_count[-1], 1, -1, dtype=self.theta.dtype
        )
        return torch.distributions.Exponential(
            lineage_count * (lineage_count - 1) / (2.0 * self.theta)
        ).rsample(sample_shape)


class PiecewiseConstantCoalescent(ConstantCoalescent):
    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        sampling_times = (
            self.sampling_times
            if node_heights.dim() == 1
            else self.sampling_times.expand(node_heights.shape[:-1] + torch.Size([-1]))
        )

        heights = torch.cat([sampling_times, node_heights], -1)
        node_mask = torch.cat(
            [
                torch.full(sampling_times.shape, 1, dtype=torch.int),  # sampling event
                torch.full(
                    sampling_times.shape[:-1]
                    + torch.Size([sampling_times.shape[-1] - 1]),
                    -1,
                    dtype=torch.int,
                ),
            ],  # coalescent event
            dim=-1,
        )

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(
            node_mask_sorted == -1,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ).cumsum(-1)[..., :-1]

        thetas = self.theta.gather(-1, thetas_indices)
        return -torch.sum(
            lchoose2 * durations / thetas, -1, keepdim=True
        ) - self.theta.log().sum(-1, keepdim=True)


class PiecewiseConstantCoalescentModel(ConstantCoalescentModel):
    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescent(self.sampling_times, self.theta.tensor)
        return pwc.log_prob(self.node_heights.tensor)


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(
        self,
        sampling_times: torch.Tensor,
        thetas: torch.Tensor,
        grid: torch.Tensor,
        validate_args=None,
    ) -> None:
        super().__init__(sampling_times, thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        if node_heights.dim() > self.theta.dim():
            batch_shape = node_heights.shape[:-1]
        else:
            batch_shape = self.theta.shape[:-1]

        sampling_times = self.sampling_times.expand(batch_shape + torch.Size([-1]))
        grid = self.grid.expand(batch_shape + torch.Size([-1]))

        if node_heights.dim() < self.theta.dim():
            heights = torch.cat(
                [
                    sampling_times,
                    node_heights.expand(batch_shape + torch.Size([-1])),
                    grid,
                ],
                -1,
            )
        else:
            heights = torch.cat([sampling_times, node_heights, grid], -1)

        node_mask = torch.cat(
            [
                torch.full(sampling_times.shape, 1, dtype=torch.int),  # sampling event
                torch.full(
                    sampling_times.shape[:-1]
                    + torch.Size([sampling_times.shape[-1] - 1]),
                    -1,
                    dtype=torch.int,
                ),
                # coalescent event
                torch.full(grid.shape, 0, dtype=torch.int),
            ],  # no event
            dim=-1,
        )

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(
            node_mask_sorted == 0,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ).cumsum(-1)

        if self.theta.dim() <= len(batch_shape):
            thetas = self.theta.expand(batch_shape + torch.Size([-1])).gather(
                -1, thetas_indices
            )
        else:
            thetas = self.theta.gather(-1, thetas_indices)

        log_thetas = torch.where(
            node_mask_sorted == -1,
            torch.log(thetas),
            torch.zeros(1, dtype=heights.dtype),
        )
        return torch.sum(
            -lchoose2 * durations / thetas[..., :-1] - log_thetas[..., 1:],
            -1,
            keepdim=True,
        )


class PiecewiseConstantCoalescentGridModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        theta: Parameter,
        node_heights: Union[Parameter, List[Parameter]],
        sampling_times: torch.Tensor,
        grid: Parameter,
    ) -> None:
        super().__init__(id_)
        self.grid = grid
        self.theta = theta
        self.sampling_times = sampling_times
        self.node_heights = node_heights
        self.add_parameter(theta)
        if isinstance(node_heights, list):
            for parameter in node_heights:
                self.add_parameter(parameter)
        else:
            self.add_parameter(node_heights)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescentGrid(
            self.sampling_times, self.theta.tensor, self.grid.tensor
        )
        if isinstance(self.node_heights, list):
            log_p = pwc.log_prob(
                torch.stack([param.tensor for param in self.node_heights])
            ).sum(0)
        else:
            log_p = pwc.log_prob(self.node_heights.tensor)
        return log_p

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']

        theta = process_object(data['theta'], dic)

        if TreeModel.tag in data:
            tree_model = process_objects(data[TreeModel.tag], dic)
            if isinstance(tree_model, list):
                sampling_times = tree_model[0].sampling_times
                node_heights = [parameter.tensor for parameter in tree_model]
            else:
                sampling_times = tree_model.sampling_times
                node_heights = tree_model._node_heights
        else:
            sampling_times, node_heights = process_times_events(
                data['times'], data['events'], theta.dtype
            )

        if 'grid' not in data:
            cutoff: float = data['cutoff']
            grid = Parameter(None, torch.linspace(0, cutoff, theta.shape[-1])[1:])
        else:
            if isinstance(data['grid'], list):
                grid = Parameter(None, torch.tensor(data['grid'], dtype=theta.dtype))
            else:
                grid = process_object(data['grid'], dic)
        assert grid.shape[0] + 1 == theta.shape[-1]

        return cls(id_, theta, node_heights, sampling_times, grid)


def process_times_events(
    time_list: list, event_list: list, dtype: torch.dtype
) -> Tuple[torch.Tensor, Parameter]:
    times = torch.tensor(time_list, dtype=dtype)
    events = torch.LongTensor(event_list)
    sampling_times = times[events == 1]
    node_heights = Parameter(None, times[events == 0])
    return sampling_times, node_heights
