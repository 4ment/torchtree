"""Integrated GMRF/gamma distribution."""
from __future__ import annotations

import math
from typing import Any

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.identifiable import Identifiable
from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..evolution.tree_model import TimeTreeModel, TreeModel
from ..typing import ID


@register_class
class GMRFGammaIntegrated(CallableModel):
    r"""Integrated GMRF/gamma distribution.

    Integrate the product of GMRF and gamma distribution with respect to
    the precision parameter. GMRF is parameterized with precision :math:`\tau`
    parameter and the gamma distribution is parameterized with shape :math:`\alpha` and
    rate :math:`\beta` parameters.

    :param id_: ID of GMRFGammaIntegrated object.
    :type id_: str or None
    :param AbstractParameter field: Markov random field.
    :param float shape: shape parameter of the gamma distribution.
    :param float rate: rate parameter of the gamma distribution.
    :param TimeTreeModel tree_model: Optional; time tree model.
        (if specified a time-aware GMRF is used).
    :param bool rescale: Optional; rescale by root height (tree_model must be specified).

    .. math::
       p(X) &= \int p(\tau; \alpha, \beta) p(X \mid \tau) d\tau \\
            &= \int \frac{\beta^\alpha}{\Gamma(\alpha)} \tau^{\alpha-1} e^{-\beta \tau} \prod_{i=1}^{N-1} \frac{1}{\sqrt{2 \pi}} \sqrt{\tau} e^{-\frac{\tau}{2} (x_{i+1} -x_i)^2} d\tau \\
            &= \left(\frac{1}{\sqrt{2 \pi}}\right)^{N-1} \frac{\beta^\alpha}{\Gamma(\alpha)} \int \tau^{\alpha + \frac{N-3}{2}} e^{-\tau (\frac{1}{2} \sum_{i=1}^{N-1} (x_{i+1} -x_i)^2 + \beta)}  d\tau\\
            &= \left(\frac{1}{\sqrt{2 \pi}}\right)^{N-1} \frac{\beta^\alpha}{\Gamma(\alpha)} \Gamma (\alpha + \frac{N-1}{2}) \left(\sum_{i=1}^{N-1} (x_{i+1} -x_i)^2 + \beta \right)^{ -\alpha - \frac{N-1}{2} }

    Idea from Cheng Zhang
    """  # noqa: E501

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
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> GMRFGammaIntegrated:
        r"""Creates a GMRFGammaIntegrated object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a
            GMRFGammaIntegrated object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - x (dict or str): Markov random field parameter.
          - shape (float): value of shape parameter of gamma distribution.
          - rate (float): value of rate parameter of gamma distribution.

         Optional:
          - tree_model (dict or str):time tree model.
          - rescale (bool): rescale by root height (Default: true).

        :example:
        >>> field = {"id": "field", "type": "Parameter", "tensor": [1., 2., 3.]}
        >>> gmrf_dic = {"id": "gmrf", "x": field, "shape": 0.1, "rate": 0.2}
        >>> gmrf = GMRFGammaIntegrated.from_json(gmrf_dic, {})
        >>> isinstance(gmrf, GMRFGammaIntegrated)
        True
        >>> isinstance(gmrf(), torch.Tensor)
        True
        >>> gmrf.id == gmrf_dic["id"]
        True

        .. note::
            If tree_model is specified the GMRF is time-aware and it should not be used
            with skygrid. The rescale parameter is ignored if tree_model is not
            specified.
        """
        id_ = data['id']
        tree_model = None
        weights = None
        # time-aware if a tree_model is provided
        if TreeModel.tag in data:
            tree_model = process_object(data[TreeModel.tag], dic)
        elif "weights" in data:
            weights = process_object(data["weights"], dic)

        x = process_object(data['x'], dic)
        shape = float(data['shape'])
        rate = float(data['rate'])
        rescale = data.get("rescale", True)

        return cls(id_, x, shape, rate, tree_model, weights, rescale)
