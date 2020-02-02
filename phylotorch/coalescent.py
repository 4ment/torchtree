import torch


class ConstantCoalescent(object):
    def __init__(self, sampling_times):
        self.sampling_times = sampling_times
        self.taxon_count = sampling_times.shape[0]

    def log_prob(self, theta, node_heights):
        heights = torch.cat([self.sampling_times, node_heights], 0)
        node_mask = torch.cat([torch.full((self.taxon_count,), False, dtype=torch.bool),
                              torch.full((self.taxon_count - 1,), True, dtype=torch.bool)], 0)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, 0, indices)
        node_mask_sorted = torch.gather(node_mask, 0, indices)
        lineage_count = torch.where(node_mask_sorted, torch.tensor([-1]), torch.tensor([1])).cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return torch.sum(-lchoose2 * durations / theta) - (self.taxon_count - 1)*torch.log(theta)


class PiecewiseConstantCoalescent(ConstantCoalescent):
    def __init__(self, sampling_times):
        super(PiecewiseConstantCoalescent, self).__init__(sampling_times)

    def log_prob(self, theta, node_heights):
        heights = torch.cat([self.sampling_times, node_heights], 0)
        node_mask = torch.cat([torch.full((self.taxon_count,), False, dtype=torch.bool),
                              torch.full((self.taxon_count - 1,), True, dtype=torch.bool)], 0)

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, 0, indices)
        node_mask_sorted = torch.gather(node_mask, 0, indices)
        lineage_count = torch.where(node_mask_sorted, torch.tensor([-1]), torch.tensor([1])).cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        masks = node_mask_sorted[1:]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas = torch.cat([torch.full((self.taxon_count - 1,), 0.0, dtype=torch.float64), theta], 0)

        thetas_masked = torch.masked_select(thetas, masks)
        return torch.sum(-torch.masked_select(lchoose2 * durations, masks) / thetas_masked - torch.log(thetas_masked))


class PiecewiseConstantCoalescentGrid(ConstantCoalescent):
    def __init__(self, sampling_times, grid):
        super(PiecewiseConstantCoalescentGrid, self).__init__(sampling_times)
        self.grid = grid

    def log_prob(self, thetas, node_heights):
        heights = torch.cat([self.sampling_times, node_heights, self.grid], 0)
        node_mask = torch.cat([torch.full((self.taxon_count,), 1),  # sampling event
                              torch.full((self.taxon_count - 1,), -1),  # coalescent event
                              torch.full((self.grid.shape[0],), 0)], 0)  # no event

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, 0, indices)
        node_mask_sorted = torch.gather(node_mask, 0, indices)
        lineage_count = node_mask_sorted.cumsum(0)[:-1]

        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0

        thetas_indices = torch.where(node_mask_sorted == 0, torch.tensor([1], dtype=torch.long),
                               torch.tensor([0], dtype=torch.long)).cumsum(0)
        thetas_masked = torch.masked_select(thetas[thetas_indices], node_mask_sorted == -1)
        return -torch.sum(lchoose2 * durations / thetas[thetas_indices[:-1]]) - torch.log(thetas_masked).sum()
