import torch
from torch.distributions import Transform

from ..core.abstractparameter import AbstractParameter
from ..core.utils import register_class
from .tree_model import TreeModel


@register_class
class LogDifferenceRateTransform(Transform):
    r"""Compute log rate difference of adjacent nodes.

    :math:`y_i = \log(r_i) - \log(r_{p(i)})`
    """

    bijective = True
    sign = +1

    def __init__(self, tree_model: TreeModel, cache_size=0) -> None:
        super().__init__(cache_size=cache_size)
        self._tree_model = tree_model

    def _call(self, x) -> torch.Tensor:
        rates = torch.cat(
            (
                x,
                torch.ones(x.shape[:-1] + (1,)),
            ),
            dim=-1,
        ).log()
        indices = self._tree_model.preorder.t()
        return rates[..., indices[1]] - rates[..., indices[0]]

    def _inverse(self, y) -> torch.Tensor:
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        return -y.sum(-1)


@register_class
class RescaledRateTransform(Transform):
    r"""Scale substitution rates

    :math:`r_i = \mu \tilde{r}_i \frac{\sum b}{\sum b r}`
    """

    bijective = True
    sign = +1

    def __init__(
        self, rate: AbstractParameter, tree_model: TreeModel, cache_size=0
    ) -> None:
        super().__init__(cache_size=cache_size)
        self._rate = rate
        self._tree_model = tree_model

    def _call(self, x) -> torch.Tensor:
        branch_lengths = self._tree_model.branch_lengths()
        return (
            self._rate.tensor
            * x
            * branch_lengths.sum(-1, keepdim=True)
            / (branch_lengths * x).sum(-1, keepdim=True)
        )

    def _inverse(self, y) -> torch.Tensor:
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        raise NotImplementedError
