import torch


class ConstantCoalescent(torch.nn.Module):
    def __init__(self, sampling_times):
        super(ConstantCoalescent, self).__init__()
        self.sampling_times = sampling_times
        self.taxon_count = sampling_times.shape[0]
        self.theta_mu = torch.nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.theta_sigma = torch.nn.Parameter(torch.randn(1, 1), requires_grad=True)

    def elbo(self, node_heights):
        m = torch.distributions.Normal(self.theta_mu, self.theta_sigma.exp())
        z = m.rsample()
        logQ = m.log_prob(z)

        theta = z.exp()

        logP = self.log_likelihood(theta, node_heights) + z  # last term is log jacobian
        logPrior = 0  # torch.log(1.0/theta)
        elbo = logP + logPrior - logQ
        return elbo

    def log_likelihood(self, theta, node_heights):
        heights = torch.cat([self.sampling_times, node_heights], 0)
        node_mask = torch.tensor([0.0] * self.taxon_count + [1.0] * (self.taxon_count - 1))

        indices = torch.argsort(heights, descending=False)
        heights_sorted = torch.gather(heights, 0, indices)
        node_mask_sorted = torch.gather(node_mask, 0, indices)
        lineage_count = torch.where(node_mask_sorted == 1, torch.tensor([-1]), torch.tensor([1])).cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        masks = node_mask_sorted[1:]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return -torch.sum(lchoose2 * durations / theta) - torch.sum(torch.log(theta) * masks)