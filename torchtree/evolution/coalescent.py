import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.parameter import Parameter
from ..core.utils import process_object, process_objects, register_class
from ..math import soft_sort
from ..typing import ID
from .tree_model import TimeTreeModel, TreeModel


class AbstractCoalescentModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        tree_model: TimeTreeModel = None,
    ) -> None:
        super().__init__(id_)
        self.theta = theta
        self.tree_model = tree_model

    def _sample_shape(self) -> torch.Size:
        return max(self.tree_model.sample_shape, self.theta.shape[:-1], key=len)


@register_class
class ConstantCoalescentModel(AbstractCoalescentModel):
    def _call(self, *args, **kwargs) -> torch.Tensor:
        coalescent = ConstantCoalescent(self.theta.tensor)
        return coalescent.log_prob(self.tree_model.node_heights)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        theta = process_object(data['theta'], dic)
        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            return cls(id_, theta, tree_model=tree_model)
        else:
            node_heights = process_data_coalesent(data, theta.dtype)
            return cls(id_, theta, FakeTreeModel(node_heights))


class AbstractCoalescentDistribution(Distribution):
    arg_constraints = {
        'theta': constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, theta: torch.Tensor, validate_args=None) -> None:
        self.theta = theta
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @classmethod
    def maximum_likelihood(cls, node_heights) -> torch.Tensor:
        raise NotImplementedError


class ConstantCoalescent(AbstractCoalescentDistribution):
    has_rsample = True

    @classmethod
    def maximum_likelihood(cls, node_heights) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, 1),
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(lchoose2 * durations) / (taxa_shape[-1] - 2)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, 1),
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(-lchoose2 * durations / self.theta, -1, keepdim=True) - (
            taxa_shape[-1] - 1
        ) * torch.log(self.theta)

    def rsample(self, sample_shape=torch.Size()):
        lineage_count = torch.arange(
            self.taxon_count[-1], 1, -1, dtype=self.theta.dtype
        )
        return torch.distributions.Exponential(
            lineage_count * (lineage_count - 1) / (2.0 * self.theta)
        ).rsample(sample_shape)


@register_class
class ExponentialCoalescentModel(AbstractCoalescentModel):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        growth: AbstractParameter,
        tree_model: TimeTreeModel = None,
    ) -> None:
        super().__init__(id_, theta, tree_model)
        self.growth = growth

    def _call(self, *args, **kwargs) -> torch.Tensor:
        coalescent = ExponentialCoalescent(self.theta.tensor, self.growth.tensor)
        return coalescent.log_prob(self.tree_model.node_heights)

    def _sample_shape(self) -> torch.Size:
        return max(
            self.tree_model.sample_shape,
            self.theta.shape[:-1],
            self.growth.shape[:-1],
            key=len,
        )

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        theta = process_object(data['theta'], dic)
        growth = process_object(data['growth'], dic)
        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            return cls(id_, theta, growth, tree_model=tree_model)
        else:
            node_heights = process_data_coalesent(data, theta.dtype)
            return cls(id_, theta, growth, FakeTreeModel(node_heights))


class ExponentialCoalescent(Distribution):
    arg_constraints = {
        'theta': constraints.positive,
        'growth': constraints.real,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(
        self, theta: torch.Tensor, growth: torch.Tensor, validate_args=None
    ) -> None:
        self.theta = theta
        self.growth = growth
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, 1),
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]
        # TODO: deal with growth==0
        height_growth_exp = torch.exp(heights_sorted * self.growth)
        integral = (height_growth_exp[..., 1:] - height_growth_exp[..., :-1]) / (
            self.theta * self.growth
        )
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        log_thetas = torch.where(
            node_mask_sorted == -1,
            torch.log(self.theta * torch.exp(-heights_sorted * self.growth)),
            torch.zeros(1, dtype=heights_sorted.dtype),
        )
        return torch.sum(-lchoose2 * integral - log_thetas[..., 1:], -1, keepdim=True)


class PiecewiseConstantCoalescent(AbstractCoalescentDistribution):
    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, 1, dtype=torch.int),  # sampling event
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                    dtype=torch.int,
                ),
            ],  # coalescent event
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
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

    @classmethod
    def maximum_likelihood(cls, node_heights: torch.Tensor) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                # sampling events
                torch.full(taxa_shape, 1, dtype=torch.int),
                # coalescent events
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                    dtype=torch.int,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]

        new_mask = node_mask_sorted.clone()
        start = 0
        for i in range(node_mask_sorted.shape[0]):
            duration = heights_sorted[i] - start
            if node_mask_sorted[i] == -1 and duration < 5.0:
                new_mask[i] = 0
            elif node_mask_sorted[i] == -1:
                start = heights_sorted[i]

        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        idx = torch.cat(
            (torch.zeros([1]), (node_mask_sorted == -1).nonzero().squeeze()), dim=0
        ).int()
        idx2 = torch.cat(
            [
                torch.full([x], idx)
                for idx, x in enumerate((idx[1:] - idx[:-1]).tolist())
            ]
        )
        thetas_hat = torch.zeros([taxa_shape[-1] - 1]).scatter_add_(
            0, idx2, lchoose2 * durations
        )
        return thetas_hat


@register_class
class PiecewiseConstantCoalescentModel(ConstantCoalescentModel):
    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescent(self.theta.tensor)
        return pwc.log_prob(self.tree_model.node_heights)


class PiecewiseConstantCoalescentGrid(AbstractCoalescentDistribution):
    def __init__(
        self,
        thetas: torch.Tensor,
        grid: torch.Tensor,
        validate_args=None,
    ) -> None:
        super().__init__(thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        batch_shape = max(node_heights.shape, self.theta.shape, key=len)[:-1]
        if node_heights.dim() > self.theta.dim():
            thetas = self.theta.expand(batch_shape + torch.Size([-1]))
        else:
            thetas = self.theta

        grid = self.grid.expand(batch_shape + torch.Size([-1]))

        if node_heights.dim() < self.theta.dim():
            heights = torch.cat(
                [
                    node_heights.expand(batch_shape + torch.Size([-1])),
                    grid,
                ],
                -1,
            )

        else:
            heights = torch.cat([node_heights, grid], -1)

        taxa_shape = heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, 1, dtype=torch.int),  # sampling event
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
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


class SoftPiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(
        self,
        thetas: torch.Tensor,
        grid: torch.Tensor,
        temperature: float = None,
        validate_args=None,
    ) -> None:
        super().__init__(thetas, validate_args)
        self.grid = grid
        self.temperature = temperature

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        batch_shape = max(node_heights.shape, self.theta.shape, key=len)[:-1]
        taxa_count = int((node_heights.shape[-1] + 1) / 2)

        if node_heights.dim() > self.theta.dim():
            thetas = self.theta.expand(batch_shape + torch.Size([-1]))
        else:
            thetas = self.theta

        grid = self.grid.expand(batch_shape + torch.Size([-1]))

        sampling_heights, sampling_counts = node_heights.flatten()[:taxa_count].unique(
            return_counts=True
        )
        sampling_heights = sampling_heights.expand(batch_shape + torch.Size([-1]))
        sampling_counts = sampling_counts.expand(batch_shape + torch.Size([-1]))

        if node_heights.dim() < self.theta.dim():
            heights = torch.cat(
                [
                    sampling_heights,
                    node_heights[..., taxa_count:].expand(
                        batch_shape + torch.Size([-1])
                    ),
                    grid,
                ],
                -1,
            )

        else:
            heights = torch.cat(
                [sampling_heights, node_heights[..., taxa_count:], grid], -1
            )

        taxa_shape = heights.shape[:-1] + (taxa_count,)
        mask_attributes = {
            'dtype': sampling_counts.dtype,
            'device': sampling_counts.device,
        }

        node_mask = torch.cat(
            [
                # sampling event
                sampling_counts,
                # coalescent event
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                    **mask_attributes,
                ),
                # no event
                torch.full(grid.shape, 0, **mask_attributes),
            ],
            dim=-1,
        )

        if self.temperature is None:
            indices = torch.argsort(heights, descending=False)
            heights_sorted = torch.gather(heights, -1, indices)
            node_mask_sorted = torch.gather(node_mask, -1, indices)
        else:
            permutation = soft_sort(-heights, self.temperature)
            heights_sorted = torch.squeeze(permutation @ heights.unsqueeze(-1), -1)
            node_mask_sorted = torch.squeeze(
                permutation @ node_mask.unsqueeze(-1).to(dtype=heights.dtype), -1
            )

        lineage_count = node_mask_sorted.cumsum(-1)[..., :-1]

        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        if self.temperature is None:
            thetas_indices = (node_mask_sorted == 0).to(dtype=torch.long).cumsum(-1)
            thetas_sorted = thetas.gather(-1, thetas_indices)
            log_thetas = torch.xlogy(node_mask_sorted == -1, thetas_sorted)
            log_p = torch.sum(
                -lchoose2 * durations / thetas_sorted[..., :-1] - log_thetas[..., 1:],
                -1,
                keepdim=True,
            )
        else:
            grid0 = torch.cat((torch.zeros(batch_shape + (1,)), grid), -1).unsqueeze(-2)
            pairwise_dist = heights_sorted[..., 1:].unsqueeze(-1) - grid0
            mat = (
                (
                    (pairwise_dist / self.temperature).cumsum(-1).softmax(-1)
                    * pairwise_dist
                )
                / self.temperature
            ).softmax(-1)
            thetas_sorted = torch.squeeze(mat @ thetas.unsqueeze(-1), -1)

            pairwise_dist = node_heights[..., taxa_count:].unsqueeze(-1) - grid0
            mat = (
                (
                    (pairwise_dist / self.temperature).cumsum(-1).softmax(-1)
                    * pairwise_dist
                )
                / self.temperature
            ).softmax(-1)
            log_thetas = torch.squeeze(mat @ thetas.unsqueeze(-1), -1).log()

            log_p = torch.sum(
                -lchoose2 * durations / thetas_sorted,
                -1,
                keepdim=True,
            ) - torch.sum(
                log_thetas,
                -1,
                keepdim=True,
            )

        return log_p


@register_class
class PiecewiseConstantCoalescentGridModel(AbstractCoalescentModel):
    def __init__(
        self,
        id_: ID,
        theta: AbstractParameter,
        grid: AbstractParameter,
        tree_model: TimeTreeModel = None,
        temperature: float = None,
    ) -> None:
        super().__init__(id_, theta)
        self.grid = grid
        self.temperature = temperature
        if isinstance(tree_model, list):
            for tree in tree_model:
                setattr(self, tree.id, tree)
        else:
            self.tree_model = tree_model

    def _call(self, *args, **kwargs) -> torch.Tensor:
        if self.temperature is not None:
            pwc = SoftPiecewiseConstantCoalescentGrid(
                self.theta.tensor, self.grid.tensor, self.temperature
            )
        else:
            pwc = PiecewiseConstantCoalescentGrid(self.theta.tensor, self.grid.tensor)
        if isinstance(self.tree_model, list):
            log_p = pwc.log_prob(
                torch.stack([model.node_heights for model in self.tree_model])
            ).sum(0)
        else:
            log_p = pwc.log_prob(self.tree_model.node_heights)
        return log_p

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']

        theta = process_object(data['theta'], dic)

        if 'grid' not in data:
            cutoff: float = data['cutoff']
            grid = Parameter(None, torch.linspace(0, cutoff, theta.shape[-1])[1:])
        else:
            if isinstance(data['grid'], list):
                grid = Parameter(None, torch.tensor(data['grid'], dtype=theta.dtype))
            else:
                grid = process_object(data['grid'], dic)
        assert grid.shape[0] + 1 == theta.shape[-1]

        optionals = {}
        if 'temperature' in data:
            optionals['temperature'] = data['temperature']

        if TreeModel.tag in data:
            tree_model = process_objects(data[TreeModel.tag], dic)
            return cls(id_, theta, grid, tree_model, **optionals)
        else:
            node_heights = process_data_coalesent(data, theta.dtype)
            return cls(id_, theta, grid, FakeTreeModel(node_heights), **optionals)


class FakeTreeModel:
    def __init__(self, node_heights):
        self._node_heights = node_heights

    @property
    def node_heights(self):
        return self._node_heights.tensor


def process_data_coalesent(data, dtype: torch.dtype) -> AbstractParameter:
    if 'times' in data:
        times = torch.tensor(data['times'], dtype=dtype)
    else:
        times = torch.tensor([0.0] + data['intervals'], dtype=dtype).cumsum(0)
    events = torch.LongTensor(data['events'])
    return Parameter(None, torch.cat((times[events == 1], times[events == 0]), -1))


class PiecewiseExponentialCoalescentGrid(Distribution):
    arg_constraints = {
        'theta': constraints.positive,
        'growth': constraints.real,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(
        self,
        theta: torch.Tensor,
        growth: torch.Tensor,
        grid: torch.Tensor,
        validate_args=None,
    ) -> None:
        self.theta = theta
        self.growth = growth
        self.grid = grid
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        if node_heights.dim() > self.theta.dim():
            batch_shape = node_heights.shape[:-1]
        else:
            batch_shape = self.theta.shape[:-1]

        grid = self.grid.expand(batch_shape + torch.Size([-1]))

        if node_heights.dim() < self.theta.dim():
            grid_heights = torch.cat(
                [
                    node_heights.expand(batch_shape + torch.Size([-1])),
                    grid,
                ],
                -1,
            )

        else:
            grid_heights = torch.cat([node_heights, grid], -1)

        if self.theta.dim() <= len(batch_shape):
            thetas = self.theta.expand(batch_shape + torch.Size([-1]))
            growth = self.growth.expand(batch_shape + torch.Size([-1]))
        else:
            thetas = self.theta
            growth = self.growth

        taxa_shape = grid_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        event_mask = torch.cat(
            [
                # sampling event
                torch.full(taxa_shape, 1, dtype=torch.int),
                # coalescent event
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    -1,
                    dtype=torch.int,
                ),
                # no event
                torch.full(grid.shape, 0, dtype=torch.int),
            ],
            dim=-1,
        )

        indices = torch.argsort(grid_heights, descending=False)
        grid_heights_sorted = torch.gather(grid_heights, -1, indices)
        event_mask_sorted = torch.gather(event_mask, -1, indices)
        lineage_count = event_mask_sorted.cumsum(-1)[..., :-1]

        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        internal_heights = node_heights[..., taxa_shape[-1] :]

        # Find indices such that grid_i <= t < grid_{i+1}
        indices_grid_heights = torch.bucketize(grid_heights_sorted, self.grid)
        indices_internals = indices_grid_heights[event_mask_sorted == -1].reshape(
            internal_heights.shape
        )

        grid0 = torch.cat((torch.zeros(batch_shape + (1,)), grid), -1)
        grid_intervals = grid0[..., 1:] - grid0[..., :-1]

        # log population size at the end of each grid point
        log_pop_size_grid = (
            torch.cat(
                (
                    torch.zeros(thetas.shape[:-1] + (1,)),
                    -torch.cumsum(growth[..., :-1] * grid_intervals, -1),
                ),
                -1,
            )
            + thetas.log()
        )

        # log population size at the end of each coalescent interval
        log_pop_sizes = log_pop_size_grid.gather(-1, indices_internals) - growth.gather(
            -1, indices_internals
        ) * (internal_heights - grid0.gather(-1, indices_internals))

        # Integrate 1/N(t) over each interval
        growth_intervals = growth.gather(-1, indices_grid_heights)
        grid_heights_growth_exp = torch.exp(grid_heights_sorted * growth_intervals)
        integral = (
            grid_heights_growth_exp[..., 1:] - grid_heights_growth_exp[..., :-1]
        ) / (thetas * growth_intervals[..., 1:])

        return -torch.sum(
            lchoose2 * integral,
            dim=-1,
            keepdim=True,
        ) - torch.sum(
            log_pop_sizes,
            dim=-1,
            keepdim=True,
        )
