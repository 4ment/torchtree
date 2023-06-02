"""Bayesian bridge prior."""
from __future__ import annotations

from numbers import Number
from typing import Any, Union

import torch
from torch import Size, Tensor

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.utils import process_object, register_class
from torchtree.typing import ID


@register_class
class BayesianBridge(CallableModel):
    """Bayesian bridge.

    [polson2014]_ and [nishimura2019]_

    :param id_: ID of object
    :param x: random variable
    :param scale: global scale
    :param alpha: exponent
    :param local_scale: local scale
    :param slab: slab width

    .. [polson2014] Polson and Scott 2014. The Bayesian Bridge.
    .. [nishimura2019] Nishimura, Suchard 2019 .Shrinkage with shrunken shoulders: Gibbs
        sampling shrinkage model posteriors with guaranteed convergence rates.
    """

    def __init__(
        self,
        id_: ID,
        x: AbstractParameter,
        scale: Union[AbstractParameter, Tensor],
        alpha: Union[AbstractParameter, Tensor] = None,
        local_scale: Union[AbstractParameter, Tensor] = None,
        slab: Union[AbstractParameter, Tensor] = None,
    ) -> None:
        super().__init__(id_)
        self.x = x
        self.scale = scale
        self.alpha = alpha
        self.local_scale = local_scale
        self.slab = slab

    def _call(self, *args, **kwargs) -> Tensor:
        global_scale = (
            self.scale.tensor
            if isinstance(self.scale, AbstractParameter)
            else self.scale
        )

        if self.local_scale is not None:
            local_scale = (
                self.local_scale.tensor
                if isinstance(self.local_scale, AbstractParameter)
                else self.local_scale
            )
            global_local = global_scale * local_scale
            if self.slab is not None:
                slab = (
                    self.slab.tensor
                    if isinstance(self.slab, AbstractParameter)
                    else self.slab
                )
                global_local /= (1.0 + (global_local / slab) ** 2).sqrt()
            return torch.distributions.Normal(
                torch.zeros_like(global_local),
                global_local,
            ).log_prob(self.x.tensor)
        else:
            alpha = (
                self.alpha.tensor
                if isinstance(self.alpha, AbstractParameter)
                else self.alpha
            )
            return -global_scale.log() - (self.x.tensor.abs() / global_scale) ** alpha

    def _sample_shape(self) -> Size:
        return self.x.tensor.shape[:-1]

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    @staticmethod
    def json_factory(id_: str, x, scale, alpha):
        model = {
            'id': id_,
            'type': 'BayesianBridge',
            'x': x,
            'scale': scale,
            'alpha': alpha,
        }
        return model

    @classmethod
    def from_json(
        cls, data: dict[str, Any], dic: dict[str, Identifiable]
    ) -> BayesianBridge:
        r"""Creates a BayesianBridge object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a
            BayesianBridge object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.

        **JSON attributes**:

         Mandatory:
          - id (str): unique string identifier.
          - x (dict or str): parameter.
          - scale (dict or str or float): global scale parameter.

         Optional:
          - alpha (dict or str or float): alpha parameter.
          - local_scale (dict or str or float): local scale parameter.
          - slab (dict or str or float): slab parameter

        :example:
        >>> x = {"id": "x", "type": "Parameter", "tensor": [1., 2., 3.]}
        >>> scale = {"id": "scale", "type": "Parameter", "tensor": [1.]}
        >>> alpha = {"id": "alpha", "type": "Parameter", "tensor": [0.1]}
        >>> bridge_dic = {"id": "bridge", "x": x, "scale": scale, "alpha": alpha}
        >>> bridge = BayesianBridge.from_json(bridge_dic, {})
        >>> isinstance(bridge, BayesianBridge)
        True
        >>> isinstance(bridge(), torch.Tensor)
        True

        .. note::
            local_scale or alpha are optional parameters but only one of them can
            be specified at a time. The slab parameter must be specified if a
            local_scale parameter is specified.
        """
        id_ = data['id']
        x = process_object(data['x'], dic)
        info = {'dtype': x.dtype, 'device': x.device}
        scale = process_object_number(data['scale'], dic, **info)

        options = {}
        if 'alpha' in data:
            options['alpha'] = process_object_number(data['alpha'], dic, **info)
        if 'local_scale' in data:
            options['local_scale'] = process_object_number(
                data['local_scale'], dic, **info
            )
            options['slab'] = process_object_number(data['slab'], dic, **info)
        return cls(id_, x, scale, **options)


def process_object_number(data, dic, **options) -> Union[Tensor, AbstractParameter]:
    """Data can be a Number, str, or dict."""
    if isinstance(data, Number):
        return torch.tensor(data, **options)
    else:
        return process_object(data, dic)
