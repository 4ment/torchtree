import math

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..evolution.tree_model import TimeTreeModel, TreeModel
from ..typing import ID


@register_class
class GMRFGammaIntegrated(CallableModel):
    r"""Integrated GMRF/gamma.

    Integrate the product of GMRF and gamma distribution with respect to
    the precision parameter. GMRF is parameterized with precision
    parameter and the gamma distributino is parameterized with shape and
    rate parameters.

    Math detivated by Chenz Zhang
    """

    def __init__(
        self,
        id_: ID,
        field: AbstractParameter,
        shape: float,
        rate: float,
        tree_model: TimeTreeModel = None,
        weights: torch.Tensor = None,
        rescale: bool = True,
    ) -> None:
        super().__init__(id_)
        self.tree_model = tree_model
        self.field = field
        self._shape = shape
        self._rate = rate
        self._weights = weights
        self._rescale = rescale
        self._dim = self.field.shape[-1] - 1.0  # field dim
        self.constant_term = (
            -self._dim / 2.0 * math.log(2.0 * math.pi)
            + self._shape * math.log(self._rate)
            - math.lgamma(self._shape)
            + math.lgamma(self._shape + self._dim / 2.0)
        )

    def _call(self, *args, **kwargs):
        diff_square = torch.pow(
            self.field.tensor[..., :-1] - self.field.tensor[..., 1:], 2.0
        )
        if self.tree_model is not None:
            heights = torch.cat(
                (
                    torch.zeros(
                        self.tree_model.node_heights.shape[:-1] + (1,),
                        dtype=self.field.dtype,
                        device=self.field.device,
                    ),
                    self.tree_model.node_heights[..., self.tree_model.taxa_count :],
                ),
                -1,
            )
            indices = torch.argsort(heights, descending=False)
            heights_sorted = torch.gather(heights, -1, indices)
            durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
            diff_square /= (durations[..., :-1] + durations[..., 1:]) / 2.0
            if self._rescale:
                diff_square *= heights_sorted[..., -1:]
        elif self._weights is not None:
            diff_square /= self._weights

        return (
            self.constant_term
            - (self._shape + self._dim / 2.0)
            * (diff_square.sum(-1, keepdim=True) / 2.0 + self._rate).log()
        )

    def _sample_shape(self) -> torch.Size:
        return self.field.tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        # time-aware if a tree_model is provided
        if TreeModel.tag in data:
            tree_model = process_object(data[TreeModel.tag], dic)
        else:
            tree_model = None
        x = process_object(data['x'], dic)
        shape = float(data['shape'])
        rate = float(data['rate'])

        return cls(id_, x, shape, rate, tree_model)
