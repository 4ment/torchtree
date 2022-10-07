import math
from numbers import Number
from typing import Union

from torch import Size, Tensor

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.model import CallableModel
from torchtree.core.utils import process_object, register_class
from torchtree.typing import ID


@register_class
class BayesianBridge(CallableModel):
    """Bayesian bridge.

    [polson2014]_

    :param id_: ID of object
    :param x: random variable
    :param scale: scale
    :param alpha: exponent

    .. [polson2014] Polson and Scott 2014. The Bayesian Bridge.
    """

    def __init__(
        self,
        id_: ID,
        x: AbstractParameter,
        scale: Union[AbstractParameter, float],
        alpha: float,
    ) -> None:
        super().__init__(id_)
        self.x = x
        self.scale = scale
        self.alpha = alpha

    def _call(self, *args, **kwargs) -> Tensor:
        if isinstance(self.scale, Number):
            return (
                -math.log(self.scale) - (self.x.tensor.abs() / self.scale) ** self.alpha
            )
        else:
            return (
                -self.scale.tensor.log()
                - (self.x.tensor.abs() / self.scale.tensor) ** self.alpha
            )

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
        if isinstance(data['scale'], Number):
            scale = data['scale']
        else:
            scale = process_object(data['scale'], dic)
        alpha = data['alpha']
        return cls(id_, x, scale, alpha)
