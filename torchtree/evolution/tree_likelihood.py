import torch
import torch.distributions

from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..typing import ID
from .branch_model import BranchModel
from .site_model import SiteModel
from .site_pattern import SitePattern
from .substitution_model.abstract import SubstitutionModel
from .tree_model import TreeModel


def calculate_treelikelihood(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Simple function for calculating the log tree likelihood.

    :param partials: list of tensors of partials [S,N] leaves and [...,S,N] internals
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,S,S]
    :param freqs: tensor of frequencies [...,S]
    :return: tree log likelihood [batch]
    """
    for node, left, right in post_indexing:
        partials[node] = (mats[..., left, :, :] @ partials[left]) * (
            mats[..., right, :, :] @ partials[right]
        )
    return torch.sum(
        torch.log(freqs @ partials[post_indexing[-1][0]]) * weights,
        -1,
    )


def calculate_treelikelihood_discrete(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood with rate categories

    number of tips: T,
    number of internal nodes: I=T-1,
    number of branches: B=2T-2,
    number of states: S,
    number of sites: N,
    number of rate categories: K.

    The shape of internal partials after peeling is [...,K,S,N].

    :param partials: list of T tip partial tensors [S,N] and I internals [None]
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


def calculate_treelikelihood_tip_states_discrete(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood with rate categories using tip states

    number of tips: T,
    number of internal nodes: I=T-1,
    number of branches: B=2T-2,
    number of states: S,
    number of sites: N,
    number of rate categories: K.

    The shape of internal partials after peeling is [...,K,S,N].

    :param partials: list of T tip state tensors [N] and I internals [None]
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """
    tip_count = len(post_indexing) + 1
    mat_tips = torch.cat(
        (
            mats[..., :tip_count, :, :, :],
            torch.ones(mats[..., :tip_count, :, :, :].shape[:-1] + (1,)),
        ),
        -1,
    )

    for node, left, right in post_indexing:
        if left < tip_count:
            p_left = mat_tips[..., left, :, :, partials[left]]
        else:
            p_left = mats[..., left, :, :, :] @ partials[left]

        if right < tip_count:
            p_right = mat_tips[..., right, :, :, partials[right]]
        else:
            p_right = mats[..., right, :, :, :] @ partials[right]

        partials[node] = p_left * p_right

    return torch.sum(
        torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], -3))
        * weights,
        -1,
    )


def calculate_treelikelihood_discrete_safe(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    r"""Calculate log tree likelihood with rate categories using rescaling.

    This function is used when an underflow is detected for the first time (i.e. inf)
    since it is not recalculating partials that are above the threshold.

    :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
    :param weights: [N]
    :param post_indexing:
    :param mats: tensor of matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :param threshold: threshold for rescaling
    :return: tree log likelihood [batch]
    """
    scalers = []
    rescaled = [False] * (post_indexing[-1][0] + 1)
    for node, left, right in post_indexing:
        if (
            rescaled[left]
            or rescaled[right]
            or torch.any(torch.max(partials[node], -2, keepdim=True)[0] < threshold)
        ):
            partial = (mats[..., left, :, :, :] @ partials[left]) * (
                mats[..., right, :, :, :] @ partials[right]
            )
            scaler, _ = torch.max(
                partial.view(*partial.shape[:-3], -1, *partial.shape[-1:]),
                -2,
                keepdim=True,
            )
            scalers.append(scaler)
            partials[node] = partial / scaler.unsqueeze(-2)
            rescaled[node] = True
    return torch.sum(
        (
            torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], dim=-3))
            + torch.cat(scalers, -2).log().sum(dim=-2).unsqueeze(-2)
        )
        * weights,
        dim=-1,
    )


def calculate_treelikelihood_discrete_rescaled(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood with rate categories using rescaling

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
        scaler, _ = torch.max(
            partial.view(*partial.shape[:-3], -1, *partial.shape[-1:]), -2, keepdim=True
        )
        scalers.append(scaler)
        partials[node] = partial / scaler.unsqueeze(-2)
    return torch.sum(
        (
            torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], dim=-3))
            + torch.cat(scalers, -2).log().sum(dim=-2).unsqueeze(-2)
        )
        * weights,
        dim=-1,
    )


def calculate_treelikelihood_tip_states_discrete_rescaled(
    partials: list,
    weights: torch.Tensor,
    post_indexing: list,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood with rate categories using tip states and rescaling

    :param partials: list of tensors of tip states [N] leaves and [...,K,S,N] internals
    :param weights: [N]
    :param post_indexing:
    :param mats: tensor of matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """
    tip_count = len(post_indexing) + 1
    mat_tips = torch.cat(
        (
            mats[..., :tip_count, :, :, :],
            torch.ones(mats[..., :tip_count, :, :, :].shape[:-1] + (1,)),
        ),
        -1,
    )

    scalers = []
    for node, left, right in post_indexing:
        if left < tip_count:
            p_left = mat_tips[..., left, :, :, partials[left]]
        else:
            p_left = mats[..., left, :, :, :] @ partials[left]

        if right < tip_count:
            p_right = mat_tips[..., right, :, :, partials[right]]
        else:
            p_right = mats[..., right, :, :, :] @ partials[right]

        partial = p_left * p_right

        scaler, _ = torch.max(
            partial.view(*partial.shape[:-3], -1, *partial.shape[-1:]), -2, keepdim=True
        )
        scalers.append(scaler)
        partials[node] = partial / scaler.unsqueeze(-2)
    return torch.sum(
        (
            torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], dim=-3))
            + torch.cat(scalers, -2).log().sum(dim=-2).unsqueeze(-2)
        )
        * weights,
        dim=-1,
    )


@register_class
class TreeLikelihoodModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        site_pattern: SitePattern,
        tree_model: TreeModel,
        subst_model: SubstitutionModel,
        site_model: SiteModel,
        clock_model: BranchModel = None,
        use_ambiguities=False,
        use_tip_states=False,
    ):
        super().__init__(id_)
        self.site_pattern = site_pattern
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        self.rescale = False
        self.use_tip_states = use_tip_states
        self.threshold = (
            1.0e-20 if subst_model.frequencies.dtype == torch.float32 else 1.0e-40
        )
        if use_tip_states:
            self.partials, self.weights = site_pattern.compute_tips_states()
        else:
            self.partials, self.weights = site_pattern.compute_tips_partials(
                use_ambiguities
            )
        self.partials.extend([None] * (len(tree_model.taxa) - 1))

    def _call(self, *args, **kwargs) -> torch.Tensor:
        branch_lengths = self.tree_model.branch_lengths()
        sample_shape = self.sample_shape
        rates = self.site_model.rates()
        # for models like JC69 rates is always tensor([1.0])  (i.e. sample_shape == [])
        if rates.dim() == 1:
            rates = rates.expand(sample_shape + (1, -1))
        else:
            rates = rates.reshape(sample_shape + (1, -1))
        probs = self.site_model.probabilities().unsqueeze(-1).unsqueeze(-1)
        if self.clock_model is None:
            bls = torch.cat(
                (
                    branch_lengths,
                    torch.zeros(
                        sample_shape + (1,),
                        dtype=branch_lengths.dtype,
                        device=branch_lengths.device,
                    ),
                ),
                -1,
            )
        else:
            if branch_lengths.dim() == 1:
                bls = self.clock_model.rates * branch_lengths.expand(
                    sample_shape + (1, -1)
                )
            else:
                bls = self.clock_model.rates * branch_lengths

        mats = self.subst_model.p_t(bls.reshape(sample_shape + (-1, 1)) * rates)
        frequencies = self.subst_model.frequencies.reshape(
            self.subst_model.frequencies.shape[:-1] + (1, -1)
        )

        if self.use_tip_states:
            log_p = self.calculate_with_tip_states(mats, frequencies, probs)
        else:
            log_p = self.calculate_with_tip_partials(mats, frequencies, probs)

        return log_p

    def calculate_with_tip_partials(self, mats, frequencies, probs):
        if self.rescale:
            log_p = calculate_treelikelihood_discrete_rescaled(
                self.partials,
                self.weights,
                self.tree_model.postorder,
                mats,
                frequencies,
                probs,
            )
        else:
            log_p = calculate_treelikelihood_discrete(
                self.partials,
                self.weights,
                self.tree_model.postorder,
                mats,
                frequencies,
                probs,
            )

            if torch.any(torch.isinf(log_p)):
                self.rescale = True
                log_p = calculate_treelikelihood_discrete_safe(
                    self.partials,
                    self.weights,
                    self.tree_model.postorder,
                    mats,
                    frequencies,
                    probs,
                    self.threshold,
                )
        return log_p

    def calculate_with_tip_states(self, mats, frequencies, probs):
        if self.rescale:
            log_p = calculate_treelikelihood_tip_states_discrete_rescaled(
                self.partials,
                self.weights,
                self.tree_model.postorder,
                mats,
                frequencies,
                probs,
            )
        else:
            log_p = calculate_treelikelihood_tip_states_discrete(
                self.partials,
                self.weights,
                self.tree_model.postorder,
                mats,
                frequencies,
                probs,
            )

            if torch.any(torch.isinf(log_p)):
                self.rescale = True
                log_p = calculate_treelikelihood_tip_states_discrete_rescaled(
                    self.partials,
                    self.weights,
                    self.tree_model.postorder,
                    mats,
                    frequencies,
                    probs,
                    self.threshold,
                )
        return log_p

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return max([model.sample_shape for model in self._models.values()], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)
        site_pattern = process_object(data[SitePattern.tag], dic)
        use_ambiguities = data.get('use_ambiguities', False)
        use_tip_states = data.get('use_tip_states', False)
        clock_model = None
        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
        return cls(
            id_,
            site_pattern,
            tree_model,
            subst_model,
            site_model,
            clock_model,
            use_ambiguities,
            use_tip_states,
        )
