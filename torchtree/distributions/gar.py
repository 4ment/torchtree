"""Gamma autoregressive model."""

from __future__ import annotations

from typing import Any

import torch

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.parameter import Parameter
from torchtree.core.utils import process_object, register_class
from torchtree.typing import ID


@register_class
class GammaAutoregressiveModel(CallableModel):
    r"""Gamma autoregressive model.

    Computes the log probability of a gamma autoregressive (GAR) model with
    :math:`x_i | x_{i-1} \sim \text{Gamma}(\alpha, \alpha / x_{i-1})`.

    The mean of :math:`x_i | x_{i-1}` is :math:`x_{i-1}` and the variance
    is :math:`x_{i-1}^2/\alpha`.

    :param id_: ID of GAR object.
    :type id_: str or None
    :param AbstractParameter x: latent state parameter.
    :param AbstractParameter shape: shape parameter.
    """

    def __init__(self, id_: ID, x: AbstractParameter, shape: AbstractParameter) -> None:
        super().__init__(id_)
        self.x = x
        self.shape = shape

    def _call(self, *args, **kwargs) -> torch.Tensor:
        return (
            torch.distributions.Gamma(
                self.shape.tensor, self.shape.tensor / self.x.tensor[..., :-1]
            )
            .log_prob(self.x.tensor[..., 1:])
            .sum(-1, keepdim=True)
        )

    def _sample_shape(self) -> torch.Size:
        return self.x.tensor.shape[:-1]

    @classmethod
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> GammaAutoregressiveModel:
        r"""Creates a gamma autoregressive model object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a gamma autoregressive model object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - x (dict or str): latent state parameter.

         Optional:
          - shape (dict or str): shape parameter. (Default: 1.0)

        :example:
        >>> x = {"id": "x", "type": "Parameter", "tensor": [1., 2., 3.]}
        >>> gar_dic = {"id": "gar", "x": x}
        >>> gar = GammaAutoregressiveModel.from_json(gar_dic, {})
        >>> isinstance(gar, GammaAutoregressiveModel)
        True
        """
        id_ = data['id']
        x = process_object(data['x'], dic)
        if 'shape' not in data:
            shape = Parameter(None, torch.tensor([1.0], dtype=x.dtype))

        else:
            shape = process_object(data['shape'], dic)
        return cls(id_, x, shape)
