"""Gaussian Markov random field priors."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributions.normal

from ..core.abstractparameter import AbstractParameter
from ..core.identifiable import Identifiable
from ..core.model import CallableModel
from ..core.parameter import Parameter
from ..core.utils import process_object, register_class
from ..evolution.tree_model import TimeTreeModel, TreeModel
from ..typing import ID


@register_class
class GMRF(CallableModel):
    r"""Gaussian Markov random field.

    GMRF is parameterized with precision :math:`\tau` parameter.

    :param id_: ID of GMRF object.
    :type id_: str or None
    :param AbstractParameter field: Markov random field parameter.
    :param AbstractParameter precision: precision parameter.
    :param TimeTreeModel tree_model: Optional; time tree model.
        (if specified a time-aware GMRF is used).
    :param bool rescale: Optional; rescale by root height
        (tree_model must be specified).

    .. math::
       p(\boldsymbol{x} \mid \tau) =  \prod_{i=1}^{N-1} \frac{1}{\sqrt{2 \pi}}
        \sqrt{\tau} e^{-\frac{\tau}{2} (x_{i+1} -x_i)^2}
    """

    def __init__(
        self,
        id_: ID,
        field: AbstractParameter,
        precision: AbstractParameter,
        tree_model: TimeTreeModel = None,
        weights: torch.Tensor = None,
        rescale: bool = True,
    ) -> None:
        super().__init__(id_)
        self.tree_model = tree_model
        self.weights = weights
        self.field = field
        self.precision = precision
        self.rescale = rescale

    def _call(self, *args, **kwargs) -> torch.Tensor:
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
            if self.rescale:
                diff_square *= heights_sorted[..., -1:]
        elif self.weights is not None:
            diff_square /= self.weights

        dim = self.field.shape[-1] - 1.0  # field dim
        precision = self.precision.tensor
        return (
            precision.log() * dim / 2.0
            - diff_square.sum(-1, keepdim=True) * precision / 2.0
            - dim / 2.0 * 1.8378770664093453
        )

    def _sample_shape(self) -> torch.Size:
        return self.field.tensor.shape[:-1]

    def precision_matrix(self) -> torch.Tensor:
        dim = self.field.shape[-1]
        precision = self.precision.tensor
        precision_matrix = torch.zeros(
            self.field.shape[:-1] + (dim, dim),
            dtype=self.field.dtype,
            device=self.field.device,
        )
        precision_matrix[..., range(dim - 1), range(1, dim)] = precision_matrix[
            ..., range(1, dim), range(dim - 1)
        ] = -precision.expand(self.field.shape[:-1] + (dim - 1,))

        precision_matrix[..., range(1, dim - 1), range(1, dim - 1)] = 2.0 * precision
        precision_matrix[..., 0, 0] = precision_matrix[
            ..., dim - 1, dim - 1
        ] = precision
        return precision_matrix

    @classmethod
    def from_json(cls, data: dict[str, Any], dic: dict[str, Identifiable]) -> GMRF:
        r"""Creates a GMRF object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a GMRF object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - x (dict or str): Markov random field parameter.
          - precision (dict or str): precision parameter.

         Optional:
          - tree_model (dict or str):time tree model.
          - rescale (bool): rescale by root height (Default: true).

        :example:
        >>> field = {"id": "field", "type": "Parameter", "tensor": [1., 2., 3.]}
        >>> precision = {"id": "precision", "type": "Parameter", "tensor": [1.]}
        >>> gmrf_dic = {"id": "gmrf", "x": field, "precision": precision}
        >>> gmrf = GMRF.from_json(gmrf_dic, {})
        >>> isinstance(gmrf, GMRF)
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
        elif 'weights' in data:
            weights = process_object(data['weights'], dic)
        field = process_object(data['x'], dic)
        precision = process_object(data['precision'], dic)
        rescale = data.get('rescale', True)
        return cls(id_, field, precision, tree_model, weights, rescale)


@register_class
class GMRFCovariate(CallableModel):
    r"""Gaussian Markov random field with covariates.

    Creates the Gaussian Markov random field with covariates prior proposed
    by\ :footcite:t:`gill2016understanding`.

    :param id_: ID of GMRF object.
    :type id_: str or None
    :param AbstractParameter field: Markov random field.
    :param AbstractParameter precision: precision parameter.
    :param AbstractParameter covariates: covariates.
    :param AbstractParameter beta: coefficients representing the effect sizes for the
        covariates.

    Let :math:`Z_{1}, \ldots , Z_{P}` be a set of :math:`\boldsymbol{Z}` predictors.
    :math:`Z_i` is observed or measured at N time points.
    :math:`x_i` is as a linear function of covariates

    .. math::
        x_i = \sum \beta_{ip} Z_{ip} + w_i

    where :math:`\boldsymbol{w}=(w_1 \ldots w_N)` is a zero-mean Gaussian process and
    :math:`\boldsymbol{\beta}=(\beta_1 \ldots \beta_N)` are coefficients.

    .. math::
        p(\boldsymbol{x} \mid \boldsymbol{Z}, \boldsymbol{\beta}, \tau)
        \propto \tau^{(N-1)/2}  e^{-\tau/2(X - \boldsymbol{Z} \boldsymbol{\beta})'
        \boldsymbol{Q} (X - \boldsymbol{Z} \boldsymbol{\beta})}

    .. footbibliography::
    """

    def __init__(
        self,
        id_: ID,
        field: AbstractParameter,
        precision: AbstractParameter,
        covariates: AbstractParameter,
        beta: AbstractParameter,
    ) -> None:
        super().__init__(id_)
        self.field = field
        self.precision = precision
        self.covariates = covariates
        self.beta = beta

    def _call(self, *args, **kwargs) -> torch.Tensor:
        dim = self.field.shape[-1]
        precision = self.precision.tensor
        covariates = (
            self.covariates.tensor
            if self.covariates.shape[:-2] == self.beta.shape[:-1]
            else self.covariates.tensor.expand(self.beta.shape[:-1], (-1,))
        )
        design_matrix = torch.zeros(
            self.field.shape[:-1] + (dim, dim),
            dtype=self.field.dtype,
            device=self.field.device,
        )
        design_matrix[..., range(dim - 1), range(1, dim)] = design_matrix[
            ..., range(1, dim), range(dim - 1)
        ] = -precision.expand(self.field.shape[:-1] + (dim - 1,))

        design_matrix[..., range(1, dim - 1), range(1, dim - 1)] = 2.0 * precision
        design_matrix[..., 0, 0] = design_matrix[..., dim - 1, dim - 1] = precision
        field_z_beta = self.field.tensor - (covariates @ self.beta.tensor)
        return (
            0.5 * (dim - 1) * precision.log()
            - 0.5 * field_z_beta.t() @ design_matrix @ field_z_beta
            - (dim - 1) / 2.0 * 1.8378770664093453
        )

    def _sample_shape(self) -> torch.Size:
        return self.field.tensor.shape[:-1]

    @classmethod
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> GMRFCovariate:
        r"""Creates a GMRFCovariate object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a GMRFCovariate
            object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - x (dict or str): Markov random field parameter.
          - precision (dict or str): precision parameter.
          - covariates (dict or str or list): covariates.
          - beta (dict or str): coefficients.

        .. note::
            If the shape of the field parameter is [...,N] and there are P covariates
            then the shape of the covariates parameter should be [N,P] and the shape
            of the beta parameter should be [...,P].
        """
        id_ = data['id']
        field = process_object(data['field'], dic)
        precision = process_object(data['precision'], dic)
        if isinstance(data['covariates'], list):
            covariates = Parameter(
                None,
                torch.tensor(
                    data['covariates'], dtype=field.dtype, device=field.device
                ),
            )
        else:
            covariates = process_object(data['covariates'], dic)
        beta = process_object(data['beta'], dic)
        return cls(id_, field, precision, covariates, beta)
