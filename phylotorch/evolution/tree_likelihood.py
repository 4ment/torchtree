from typing import Optional

import torch
import torch.distributions

from ..core.model import CallableModel
from ..core.utils import process_object
from .branch_model import BranchModel
from .site_model import SiteModel
from .site_pattern import SitePattern
from .substitution_model import SubstitutionModel
from .tree_model import TreeModel


def calculate_treelikelihood(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Simple function for calculating the log tree likelihood.

    :param partials: list of tensors of partials [S,N] leaves and [S,N] internals
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [B,S,S]
    :param freqs: tensor of frequencies [S]
    :return: tree log likelihood [1]
    """
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(
            mats[right], partials[right]
        )
    return torch.sum(
        torch.log(torch.matmul(freqs, partials[post_indexing[-1][0]])) * weights
    )


def calculate_treelikelihood_discrete(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood

    :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """
    for node, left, right in post_indexing:
        partials[node] = (mats[..., left, :, :, :] @ partials[left]) * (
            mats[..., right, :, :, :] @ partials[right]
        )
    return torch.sum(
        torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], -3))
        * weights,
        -1,
    )


def calculate_treelikelihood_discrete_rescaled(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood using rescaling

    :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
    :param weights: [N]
    :param post_indexing:
    :param mats: tensor of matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """
    scalers = []
    for node, left, right in post_indexing:
        partial = (mats[..., left, :, :, :] @ partials[left]) * (
            mats[..., right, :, :, :] @ partials[right]
        )
        scaler, _ = torch.max(partial, -2)
        scalers.append(scaler)
        partials[node] = partial / scaler
    return torch.sum(
        (
            torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], dim=-3))
            + torch.sum(torch.cat(scalers, -2).log(), dim=-2)
        )
        * weights,
        dim=-1,
    )


class TreeLikelihoodModel(CallableModel):
    def __init__(
        self,
        id_: Optional[str],
        site_pattern: SitePattern,
        tree_model: TreeModel,
        subst_model: SubstitutionModel,
        site_model: SiteModel,
        clock_model: BranchModel = None,
    ):
        super().__init__(id_)
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
        self.rescale = False

    def _call(self, *args, **kwargs) -> torch.Tensor:
        branch_lengths = self.tree_model.branch_lengths()
        batch_shape = self.site_model.rates().shape[:-1]
        rates = self.site_model.rates()
        # for models like JC69 rates is always tensor([1.0])  (i.e. batch_shape == [])
        if rates.dim() == 1:
            rates = rates.expand(batch_shape + (1, -1))
        else:
            rates = rates.reshape(batch_shape + (1, -1))
        probs = self.site_model.probabilities().reshape((-1, 1, 1))

        if self.clock_model is None:
            bls = torch.cat(
                (
                    branch_lengths,
                    torch.zeros(batch_shape + (1,), dtype=branch_lengths.dtype),
                ),
                -1,
            )
        else:
            if branch_lengths.dim() == 1:
                bls = self.clock_model.rates * branch_lengths.expand(
                    batch_shape + (1, -1)
                )
            else:
                bls = self.clock_model.rates * branch_lengths

        mats = self.subst_model.p_t(bls.reshape(batch_shape + (-1, 1)) * rates)
        frequencies = self.subst_model.frequencies
        if self.rescale:
            log_p = calculate_treelikelihood_discrete_rescaled(
                self.site_pattern.partials,
                self.site_pattern.weights,
                self.tree_model.postorder,
                mats,
                frequencies.reshape(frequencies.shape[:-1] + (1, -1)),
                probs,
            )
        else:
            log_p = calculate_treelikelihood_discrete(
                self.site_pattern.partials,
                self.site_pattern.weights,
                self.tree_model.postorder,
                mats,
                frequencies.reshape(frequencies.shape[:-1] + (1, -1)),
                probs,
            )
        return log_p

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def batch_shape(self) -> torch.Size:
        return self.tree_model.batch_shape

    @property
    def sample_shape(self) -> torch.Size:
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
