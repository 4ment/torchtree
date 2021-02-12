import numpy as np
import torch
import torch.distributions

from .tree import transform_ratios
# from .transforms import NodeHeightAutogradFunction
from ..core.model import CallableModel
from ..core.utils import process_object


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


def calculate_treelikelihood_discrete_rescaled(partials, weights, post_indexing, mats, freqs, props):
    r""" Calculate log likelihood

    :param partials: list of tensors of partials [S,N] leaves and [K,S,N] internals
    :param weights: [N]
    :param post_indexing:
    :param mats: tensor of matrices [B,K,S,S]
    :param freqs: tensor of frequencies [S]
    :param props: tensor of proportions [K,1,1]
    :return:
    """
    scalers = []
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(
            mats[right], partials[right])
        scaler = torch.max(partials[node])
        partials[node] = partials[node].clone() / scaler
        scalers.append(torch.unsqueeze(scaler, 0))
    return torch.sum(
        torch.log(torch.matmul(freqs, torch.sum(props * partials[post_indexing[-1][0]], 0))) * weights) + torch.sum(
        torch.cat(scalers).log())


class TreeLikelihood(object):
    def __init__(self, id_, partials, weights, tree_model, subst_model, site_model, clock_model=None):
        self.partials = partials
        self.weights = weights
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        super().__init__(id_)

    def log_prob(self, value):
        self.site_model.update(value)
        rates = self.site_model.rates().reshape(1, -1)
        probs = self.site_model.probabilities().reshape((-1, 1, 1))

        self.tree_model.update(value)
        if self.clock_model is None:
            bls = torch.unsqueeze(
                torch.cat((self.tree_model.branch_lengths(), torch.zeros(1, dtype=torch.float64))).reshape(-1,
                                                                                                           1) * rates,
                -1)
        else:
            self.clock_model.update(value)
            clock_rates = self.clock_model.rates().reshape(-1, 1)
            bls = torch.unsqueeze(clock_rates * self.tree_model.branch_lengths().reshape(-1, 1) * rates, -1)
        self.subst_model.update(value)
        mats = self.subst_model.p_t(bls)

        return calculate_treelikelihood_discrete(self.partials, self.weights, self.tree_model.postorder, mats,
                                                 self.subst_model.frequencies, probs)


class TreeLikelihoodModel(CallableModel):

    def __init__(self, id_, site_pattern, tree_model, subst_model, site_model, clock_model=None):
        self.site_pattern = site_pattern
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        super(TreeLikelihoodModel, self).__init__(id_)

    def log_prob(self, value):
        self.site_model.update(value)
        rates = self.site_model.rates().reshape(1, -1)
        probs = self.site_model.probabilities().reshape((-1, 1, 1))

        self.tree_model.update(value)
        if self.clock_model is None:
            bls = torch.unsqueeze(
                torch.cat((self.tree_model.branch_lengths(), torch.zeros(1, dtype=torch.float64))).reshape(-1,
                                                                                                           1) * rates,
                -1)
        else:
            self.clock_model.update(value)
            clock_rates = self.clock_model.rates().reshape(-1, 1)
            bls = torch.unsqueeze(clock_rates * self.tree_model.branch_lengths().reshape(-1, 1) * rates, -1)
        self.subst_model.update(value)
        mats = self.subst_model.p_t(bls)

        return calculate_treelikelihood_discrete(self.site_pattern.partials, self.site_pattern.weights,
                                                 self.tree_model.postorder, mats,
                                                 self.subst_model.frequencies, probs)

    def __call__(self):
        rates = self.site_model.rates().reshape(1, -1)
        probs = self.site_model.probabilities().reshape((-1, 1, 1))

        if self.clock_model is None:
            bls = torch.unsqueeze(
                torch.cat((self.tree_model.branch_lengths(), torch.zeros(1, dtype=torch.float64))).reshape(-1,
                                                                                                           1) * rates,
                -1)
        else:
            clock_rates = self.clock_model.rates.reshape(-1, 1)
            bls = torch.unsqueeze(clock_rates * self.tree_model.branch_lengths().reshape(-1, 1) * rates, -1)
        mats = self.subst_model.p_t(bls)

        return calculate_treelikelihood_discrete(self.site_pattern.partials, self.site_pattern.weights,
                                                 self.tree_model.postorder, mats,
                                                 self.subst_model.frequencies, probs)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data['tree'], dic)
        site_model = process_object(data['sitemodel'], dic)
        subst_model = process_object(data['substitutionmodel'], dic)
        site_pattern = process_object(data['sitepattern'], dic)
        clock_model = None
        if 'clockmodel' in data:
            clock_model = process_object(data['clockmodel'], dic)
        return cls(id_, site_pattern, tree_model, subst_model, site_model, clock_model)


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
    # node_heights = NodeHeightAutogradFunction.apply(torch.cat((ratios, root_height)), bounds, pre_indexing, post_indexing)
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
