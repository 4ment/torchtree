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
        temp = node_mask_sorted[1:] | node_mask_sorted[:-1]
        durations = torch.masked_select(heights_sorted[1:] - heights_sorted[:-1], temp)
        lchoose2 = torch.masked_select(lineage_count * (lineage_count - 1) / 2.0, temp)
        return torch.sum(-lchoose2 * durations / theta - torch.log(theta).repeat(self.taxon_count - 1))


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

        thetas = torch.cat([torch.full((self.taxon_count - 1,), 0.0), theta], 0)

        thetas_masked = torch.masked_select(thetas, masks)
        return torch.sum(-torch.masked_select(lchoose2 * durations, masks) / thetas_masked - torch.log(thetas_masked))