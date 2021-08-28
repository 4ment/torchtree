import torch
from torch.distributions import Transform

from .tree_model import TimeTreeModel


class DifferenceNodeHeightTransform(Transform):
    r"""Transform from node height differences to node heights.

    The height :math:`x_i` of node :math:`i` is parameterized as

    .. math::

      x_i = \max(x_{c(i,0)}, x_{c(i,1)}) + \exp(y_i)

    where :math:`x_c(i,j)` is the height of the jth child of node :math:`i` and
    :math:`y_i \in \mathbb{R}`.
    """
    bijective = True
    sign = +1

    def __init__(self, tree: TimeTreeModel, cache_size=0) -> None:
        super().__init__(cache_size=cache_size)
        self.tree = tree
        self.taxa_count = self.tree.taxa_count

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Transform node height differences to internal node heights."""
        heights = list(
            self.tree.sampling_times.expand(x.shape[:-1] + (-1,)).split(1, -1)
        ) + [None] * (self.taxa_count - 1)
        x_exp = x.exp()
        for node, left, right in self.tree.postorder:
            heights[node] = (
                torch.max(heights[left], heights[right])
                + x_exp[..., node - self.taxa_count : (node - self.taxa_count + 1)]
            )
        return torch.cat(heights[self.taxa_count :], -1)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Transform internal node heights to height differences."""
        x = [None] * (self.taxa_count - 1)
        heights = list(
            self.tree.sampling_times.expand(y.shape[:-1] + (-1,)).split(1, -1)
        ) + list(y.split(1, -1))
        for node, left, right in self.tree.postorder:
            x[node - self.taxa_count] = heights[node] - torch.max(
                heights[left], heights[right]
            )
        return torch.cat(x, -1).log()

    def log_abs_det_jacobian(self, x, y):
        return x.sum(-1)
