import torch
from torch.distributions import Transform

from ..math import soft_max


class GeneralNodeHeightTransform(Transform):
    r"""Transform from ratios to node heights."""
    bijective = True
    sign = +1

    def __init__(self, tree: 'TimeTreeModel', cache_size=0) -> None:  # noqa: F821
        super().__init__(cache_size=cache_size)
        self.tree = tree
        self.taxa_count = self.tree.taxa_count
        self._indices = None
        self._det_indices = None
        self._forward_indices = None
        self._bounds = None
        self.update_bounds()
        self.sort_indices()

    def sort_indices(self):
        self._forward_indices = (
            self.tree.preorder[self.tree.preorder[..., 1] >= self.taxa_count, :]
            - self.taxa_count
        )
        self._det_indices = (
            self.tree.preorder[torch.argsort(self.tree.preorder[..., 1])].t()[
                0, self.taxa_count :
            ]
            - self.taxa_count
        )

    def update_bounds(self) -> None:
        """Called when topology changes."""
        taxa_count = self.taxa_count
        internal_heights = [None] * (taxa_count - 1)
        for node, left, right in self.tree.postorder:
            left_height = (
                self.tree.sampling_times[left]
                if left < taxa_count
                else internal_heights[left - taxa_count]
            )
            right_height = (
                self.tree.sampling_times[right]
                if right < taxa_count
                else internal_heights[right - taxa_count]
            )

            internal_heights[node - taxa_count] = (
                left_height if left_height > right_height else right_height
            )
        self._bounds = torch.cat(
            (self.tree.sampling_times, torch.stack(internal_heights)), -1
        )

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Transform node ratios and root height to internal node heights."""
        heights = x.clone()
        bounds = self._bounds[self.taxa_count :]
        for parent_id, id_ in self._forward_indices:
            heights[..., id_] = bounds[id_] + x[..., id_] * (
                heights[..., parent_id] - bounds[id_]
            )
        return heights

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Transform internal node heights to ratios/root height."""
        indices = self.tree.preorder[torch.argsort(self.tree.preorder[:, 1])].t()
        bounds = self._bounds[indices[1, self.taxa_count :]]
        return torch.cat(
            (
                (
                    y[
                        ...,
                        indices[1, self.taxa_count :] - self.taxa_count,
                    ]
                    - bounds
                )
                / (
                    y[
                        ...,
                        indices[0, self.taxa_count :] - self.taxa_count,
                    ]
                    - bounds
                ),
                y[..., -1:],
            )
        )

    def log_abs_det_jacobian(self, x, y):
        return torch.log(
            y[..., self._det_indices] - self._bounds[self.taxa_count : -1]
        ).sum(-1)


class DifferenceNodeHeightTransform(Transform):
    r"""Transform from node height differences to node heights.

    The height :math:`x_i` of node :math:`i` is parameterized as

    .. math::

      x_i = \max(x_{c(i,0)}, x_{c(i,1)}) + y_i

    where :math:`x_c(i,j)` is the height of the jth child of node :math:`i` and
    :math:`y_i \in \mathbb{R}^+`. Function max can be approximated using logsumexp
    in order to propagate the gradient if k > 0.
    """
    bijective = True
    sign = +1

    def __init__(
        self, tree_model: 'TimeTreeModel', k: float = 1.0, cache_size=0  # noqa: F821
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.tree = tree_model
        self.taxa_count = self.tree.taxa_count
        self.k = k
        if self.k <= 0:
            self.max = lambda input: torch.max(input, dim=-1, keepdim=True)[0]
        else:
            self.max = lambda input: soft_max(input, self.k, dim=-1, keepdim=True)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Transform node height differences to internal node heights."""
        heights = list(
            self.tree.sampling_times.expand(x.shape[:-1] + (-1,)).split(1, -1)
        ) + [None] * (self.taxa_count - 1)
        for node, left, right in self.tree.postorder:
            heights[node] = (
                self.max(torch.cat((heights[left], heights[right]), -1))
                + x[..., node - self.taxa_count : (node - self.taxa_count + 1)]
            )
        return torch.cat(heights[self.taxa_count :], -1)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Transform internal node heights to height differences."""
        x = [None] * (self.taxa_count - 1)
        heights = list(
            self.tree.sampling_times.expand(y.shape[:-1] + (-1,)).split(1, -1)
        ) + list(y.split(1, -1))
        for node, left, right in self.tree.postorder:
            x[node - self.taxa_count] = (
                heights[node]
                - torch.logsumexp(
                    torch.cat((heights[left] * self.k, heights[right] * self.k), -1),
                    dim=-1,
                    keepdim=True,
                )
                / self.k
            )  # - torch.max(heights[left], heights[right])
        return torch.cat(x, -1)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[:-1])
