import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution


class ConstantCoalescent(Distribution):
    arg_constraints = {'theta': constraints.positive,
                       'sampling_times': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, sampling_times, theta, validate_args=None):
        self.sampling_times = sampling_times
        self.theta = theta
        self.taxon_count = sampling_times.shape
        batch_shape, event_shape = self.theta.shape[:-1], self.theta.shape[-1:]
        super(ConstantCoalescent, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, node_heights):
        heights = torch.cat([self.sampling_times, node_heights], dim=-1)
        node_mask = torch.cat([torch.full(self.taxon_count, False, dtype=torch.bool),
                               torch.full(self.taxon_count[:-1] + (self.taxon_count[-1] - 1,), True, dtype=torch.bool)],
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = torch.where(node_mask_sorted, torch.tensor([-1]), torch.tensor([1])).cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(-lchoose2 * durations / self.theta) - (self.taxon_count[-1] - 1) * torch.log(self.theta)

    def rsample(self, sample_shape=torch.Size()):
        lineage_count = torch.arange(self.taxon_count[-1], 1, -1, dtype=torch.float64)
        return torch.distributions.Exponential(lineage_count * (lineage_count - 1) / (2.0 * self.theta)).rsample(
            sample_shape)


class PiecewiseConstantCoalescent(ConstantCoalescent):
    def __init__(self, sampling_times, thetas, validate_args=None):
        super(PiecewiseConstantCoalescent, self).__init__(sampling_times, thetas, validate_args)

    def log_prob(self, node_heights):
        heights = torch.cat([self.sampling_times, node_heights], dim=-1)
        node_mask = torch.cat([torch.full(self.taxon_count, False, dtype=torch.bool),
                               torch.full(self.taxon_count[:-1] + (self.taxon_count[-1] - 1,), True, dtype=torch.bool)],
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = torch.where(node_mask_sorted, torch.tensor([-1]), torch.tensor([1])).cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(node_mask_sorted, torch.tensor([1], dtype=torch.long),
                                     torch.tensor([0], dtype=torch.long)).cumsum(0)[:-1]
        thetas_masked = torch.masked_select(self.theta[thetas_indices], node_mask_sorted[1:])

        return -torch.sum(lchoose2 * durations / self.theta[thetas_indices]) - torch.log(thetas_masked).sum()


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(self, sampling_times, thetas, grid, validate_args=None):
        super(PiecewiseConstantCoalescentGrid, self).__init__(sampling_times, thetas, validate_args)
        self.grid = grid

    def log_prob(self, node_heights):
        heights = torch.cat([self.sampling_times, node_heights, self.grid], -1)
        node_mask = torch.cat([torch.full(self.taxon_count, 1),  # sampling event
                               torch.full(self.taxon_count[:-1] + (self.taxon_count[-1] - 1,), -1),  # coalescent event
                               torch.full((self.grid.shape[0],), 0)],  # no event
                              dim=-1)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, -1, indices)
        node_mask_sorted = torch.gather(node_mask, -1, indices)
        lineage_count = node_mask_sorted.cumsum(0)[:-1]

        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(node_mask_sorted == 0, torch.tensor([1], dtype=torch.long),
                                     torch.tensor([0], dtype=torch.long)).cumsum(0)
        thetas_masked = torch.masked_select(self.theta[thetas_indices], node_mask_sorted == -1)
        return -torch.sum(lchoose2 * durations / self.theta[thetas_indices[:-1]]) - torch.log(thetas_masked).sum()
