import numpy as np
import torch
import torch.distributions

from .tree import transform_ratios, BranchLengthTransform


def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs):
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(mats[right], partials[right])
    return torch.sum(torch.log(torch.matmul(freqs, partials[post_indexing[-1][0]])) * weights)


def calculate_treelikelihood_discrete(partials, weights, post_indexing, mats, freqs, props):
    r""" Calculate log likelihood

    :param partials: list of tensors of partials [S,N] leaves and [K,S,N] internals
    :param weights: [N]
    :param post_indexing:
    :param mats: tensor of matrices [B,K,S,S]
    :param freqs: tensor of frequencies [S]
    :param props: tensor of proportions [K,1,1]
    :return:
    """
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(
            mats[right], partials[right])
    return torch.sum(torch.log(torch.matmul(freqs, torch.sum(props * partials[post_indexing[-1][0]], 0))) * weights)


class TreeLikelihood(object):
    def __init__(self, partials, weights, bounds, pre_indexing, post_indexing, subst_model, site_model):
        self.partials = partials
        self.weights = weights
        self.bounds = bounds
        self.pre_indexing = pre_indexing
        self.post_indexing = post_indexing
        self.subst_model = subst_model
        self.site_model = site_model
        self.branch_transform = BranchLengthTransform(bounds, pre_indexing)

    def log_prob(self, branch_lengths, clock=None, subst_rates=None, subst_freqs=None, site_model=None):

        # could be a constant site model or shape is fixed
        if weibull_shape is not None:
            site_model.update(weibull_shape)

        rates = site_model.rates().reshape(1, -1)
        probs = site_model.probabilities().reshape((-1, 1, 1))

        if clock is None:
            bls = torch.unsqueeze(branch_lengths.reshape(-1, 1) * rates, -1)
        else:
            bls = torch.unsqueeze(clock * self.branch_transform(branch_lengths).reshape(-1, 1) * rates, -1)

        if subst_rates is not None or subst_freqs is not None:
            self.subst_model.update((subst_rates, subst_freqs))
        mats = self.subst_model.p_t(bls)

        return calculate_treelikelihood_discrete(self.partials, self.weights, self.post_indexing, mats,
                                                 self.subst_model.frequencies, probs)


def pytorch_likelihood(subst_model, site_model, partials, weights, bounds, pre_indexing, post_indexing,
                       root_height, ratios, clock, weibull_shape, subst_rates, subst_freqs):
    taxa_count = ratios.shape[0] + 2

    node_heights = transform_ratios(torch.cat((ratios, root_height)), bounds, pre_indexing)
    indices_sorted = pre_indexing[np.argsort(pre_indexing[:, 1])].transpose()
    branch_lengths = node_heights[indices_sorted[0, :] - taxa_count] - torch.cat(
        (bounds[:taxa_count], node_heights[indices_sorted[1, taxa_count:] - taxa_count]))

    log_det_jacobian = torch.log(
        node_heights[indices_sorted[0, taxa_count:] - taxa_count] - bounds[taxa_count:-1]).sum()

    # could be a constant site model or shape is fixed
    if weibull_shape is not None:
        site_model.update(weibull_shape)

    rates = site_model.rates().reshape(1, -1)
    probs = site_model.probabilities().reshape((-1, 1, 1))

    bls = torch.unsqueeze(clock * branch_lengths.reshape(-1, 1) * rates, -1)

    if subst_rates is not None or subst_freqs is not None:
        subst_model.update((subst_rates, subst_freqs))

    mats = subst_model.p_t(bls)
    log_p = calculate_treelikelihood_discrete(partials, weights, post_indexing, mats, subst_model.frequencies, probs)
    log_p += torch.squeeze(log_det_jacobian)
    return log_p, node_heights


def pytorch_likelihood_constant(subst_model, partials, weights, bounds, pre_indexing, post_indexing, root_height,
                                ratios, clock):
    taxa_count = ratios.shape[0] + 2

    node_heights = transform_ratios(torch.cat((ratios, root_height)), bounds, pre_indexing)
    indices_sorted = pre_indexing[np.argsort(pre_indexing[:, 1])].transpose()
    branch_lengths = node_heights[indices_sorted[0, :] - taxa_count] - torch.cat(
        (bounds[:taxa_count], node_heights[indices_sorted[1, taxa_count:] - taxa_count]))

    log_det_jacobian = torch.log(
        node_heights[indices_sorted[0, taxa_count:] - taxa_count] - bounds[taxa_count:-1]).sum()

    mats = subst_model.p_t(branch_lengths * clock)
    log_p = calculate_treelikelihood(partials, weights, post_indexing, mats, subst_model.frequencies)
    log_p += torch.squeeze(log_det_jacobian)
    return log_p, node_heights
