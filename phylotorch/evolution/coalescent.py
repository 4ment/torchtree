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
        *,
        tree_model: TimeTreeModel = None,
        node_heights: Parameter = None,
    ) -> None:
        super().__init__(id_)
        self.theta = theta
        self.add_parameter(theta)
        self.tree_model = tree_model
        if tree_model:
            self.add_model(tree_model)
        self.node_heights = node_heights
        if node_heights:
            self.add_parameter(node_heights)

    def handle_model_changed(self, model, obj, index):
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    def _call(self, *args, **kwargs) -> torch.Tensor:
        coalescent = ConstantCoalescent(self.theta.tensor)
        if self.tree_model:
            return coalescent.log_prob(self.tree_model.node_heights)
        else:
            return coalescent.log_prob(self.node_heights.tensor)

    @property
    def sample_shape(self) -> torch.Size:
        if self.tree_model:
            return max(
                self.tree_model.node_heights.shape[:-1], self.theta.shape[:-1], key=len
            )
        return max(
            [parameter.shape[:-1] for parameter in self._parameters],
            key=len,
        )

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        theta = process_object(data['theta'], dic)
        if TreeModel.tag in data:
            tree_model: TimeTreeModel = process_object(data[TreeModel.tag], dic)
            return cls(id_, theta, tree_model=tree_model)
        else:
            node_heights = process_data_coalesent(data, theta.dtype)
            return cls(id_, theta, node_heights=node_heights)


class ConstantCoalescent(Distribution):

    arg_constraints = {
        'theta': constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, theta: torch.Tensor, validate_args=None) -> None:
        self.theta = theta
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = torch.cat(
            [
                torch.full(taxa_shape, False, dtype=torch.bool),
                torch.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    True,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )

        indices = torch.argsort(node_heights, descending=False)
        heights_sorted = torch.gather(node_heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = torch.where(
            node_mask_sorted,
            torch.full_like(self.theta, -1),
            torch.full_like(self.theta, 1),
        ).cumsum(-1)[..., :-1]

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


class PiecewiseConstantCoalescent(ConstantCoalescent):
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


class PiecewiseConstantCoalescentModel(ConstantCoalescentModel):
    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescent(self.theta.tensor)
        if self.tree_model:
            return pwc.log_prob(self.tree_model.node_heights)
        else:
            return pwc.log_prob(self.node_heights.tensor)


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(
        self,
        thetas: torch.Tensor,
        grid: torch.Tensor,
        validate_args=None,
    ) -> None:
        super().__init__(thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights: torch.Tensor) -> torch.Tensor:
        if node_heights.dim() > self.theta.dim():
            batch_shape = node_heights.shape[:-1]
        else:
            batch_shape = self.theta.shape[:-1]

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


class PiecewiseConstantCoalescentGridModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        theta: Parameter,
        grid: Parameter,
        *,
        tree_model: TimeTreeModel = None,
        node_heights: Parameter = None,
    ) -> None:
        super().__init__(id_)
        self.grid = grid
        self.theta = theta
        self.add_parameter(theta)
        self.tree_model = tree_model
        self.node_heights = node_heights
        if tree_model:
            if isinstance(tree_model, list):
                for tree in tree_model:
                    self.add_model(tree)
            else:
                self.add_model(tree_model)
        else:
            self.add_parameter(node_heights)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        pwc = PiecewiseConstantCoalescentGrid(self.theta.tensor, self.grid.tensor)
        if isinstance(self.tree_model, list):
            log_p = pwc.log_prob(
                torch.stack([model.node_heights for model in self.tree_model])
            ).sum(0)
        elif self.tree_model:
            log_p = pwc.log_prob(self.tree_model.node_heights)
        else:
            log_p = pwc.log_prob(self.node_heights.tensor)
        return log_p

    def handle_model_changed(self, model, obj, index) -> None:
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        if self.tree_model:
            return max(
                self.tree_model.node_heights.shape[:-1], self.theta.shape[:-1], key=len
            )
        return max(
            [parameter.shape[:-1] for parameter in self._parameters],
            key=len,
        )

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

        if TreeModel.tag in data:
            tree_model = process_objects(data[TreeModel.tag], dic)
            return cls(id_, theta, grid, tree_model=tree_model)
        else:
            node_heights = process_data_coalesent(data, theta.dtype)
            return cls(id_, theta, grid, node_heights=node_heights)


def process_data_coalesent(data, dtype: torch.dtype) -> Parameter:
    if 'times' in data:
        times = torch.tensor(data['times'], dtype=dtype)
    else:
        times = torch.tensor([0.0] + data['intervals'], dtype=dtype).cumsum(0)
    events = torch.LongTensor(data['events'])
    return Parameter(None, torch.cat((times[events == 1], times[events == 0]), -1))
