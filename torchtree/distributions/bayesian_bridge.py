from numbers import Number
from typing import Union

import torch
from torch import Size, Tensor

from torchtree.core.abstractparameter import AbstractParameter
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

    @property
    def sample_shape(self) -> Size:
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
    def from_json(cls, data, dic):
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
    """data can be a Number, str, or dict."""
    if isinstance(data, Number):
        return torch.tensor(data, **options)
    else:
        return process_object(data, dic)
