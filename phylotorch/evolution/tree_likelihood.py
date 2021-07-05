from typing import Optional

import torch
import torch.distributions

from .branch_model import BranchModel
from .site_model import SiteModel
from .site_pattern import SitePattern
from .substitution_model import SubstitutionModel
from .tree_model import TreeModel
from ..core.model import CallableModel
from ..core.utils import process_object


def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs):
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(mats[right], partials[right])
    return torch.sum(torch.log(torch.matmul(freqs, partials[post_indexing[-1][0]])) * weights)


def calculate_treelikelihood_discrete(partials, weights, post_indexing, mats, freqs, props):
    r""" Calculate log likelihood

    :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,S]
    :param props: tensor of proportions [...,K,1,1]
    :return:
    """
    for node, left, right in post_indexing:
        partials[node] = (mats[..., left, :, :, :] @ partials[left]) * (mats[..., right, :, :, :] @ partials[right])
    return torch.sum(torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], -3)) * weights, -1)


def calculate_treelikelihood_discrete_rescaled(partials, weights, post_indexing, mats, freqs, props):
    # TODO: adapt to calculate using batch samples
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


class TreeLikelihoodModel(CallableModel):

    def __init__(self, id_: Optional[str], site_pattern: SitePattern, tree_model: TreeModel,
                 subst_model: SubstitutionModel,
                 site_model: SiteModel, clock_model: BranchModel = None):
        super(TreeLikelihoodModel, self).__init__(id_)
        self.site_pattern = site_pattern
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        self.add_model(tree_model)
        self.add_model(subst_model)
        self.add_model(site_model)
        if clock_model:
            self.add_model(clock_model)

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

    def _call(self):
        branch_lengths = self.tree_model.branch_lengths()
        batch_shape = branch_lengths.shape[:-1]
        rates = self.site_model.rates()
        # for models like JC69 rates is always tensor([1.0])  (i.e. batch_shape == [])
        if rates.dim() == 1:
            rates = rates.expand(batch_shape + (1, -1))
        else:
            rates = rates.reshape(batch_shape + (1, -1))
        probs = self.site_model.probabilities().reshape((-1, 1, 1))

        if self.clock_model is None:
            bls = torch.cat((branch_lengths, torch.zeros(batch_shape + (1,), dtype=branch_lengths.dtype)), -1)
        else:
            clock_rates = self.clock_model.rates
            bls = clock_rates * branch_lengths

        mats = self.subst_model.p_t(bls.reshape(batch_shape + (-1, 1)) * rates)
        frequencies = self.subst_model.frequencies

        return calculate_treelikelihood_discrete(self.site_pattern.partials,
                                                 self.site_pattern.weights,
                                                 self.tree_model.postorder,
                                                 mats,
                                                 frequencies.reshape(frequencies.shape[:-1] + (1, -1)),
                                                 probs)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def batch_shape(self):
        return self.tree_model.batch_shape

    @property
    def sample_shape(self):
        return self.tree_model.sample_shape

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)
        site_pattern = process_object(data[SitePattern.tag], dic)
        clock_model = None
        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
        return cls(id_, site_pattern, tree_model, subst_model, site_model, clock_model)
