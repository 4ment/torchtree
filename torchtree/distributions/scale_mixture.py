from typing import Union

from torch import Size, Tensor
from torch.distributions import Normal

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.model import CallableModel
from torchtree.core.utils import process_object, register_class
from torchtree.typing import ID


@register_class
class ScaleMixtureNormal(CallableModel):
    """Scale mixture of Normal distributions.

    Regularized when a slab width parameter or scalar is provided [piironen2017]_

    :param id_: ID of object
    :param loc: mean of the distribution
    :param x: random variable
    :param scale: global scale
    :param gamma: local scale
    :param slab: slab width

    .. [piironen2017] Piironen and Vehtari 2017. Sparsity information and regularization
     in the horseshoe and other shrinkage priors.
    """

    def __init__(
        self,
        id_: ID,
        x: AbstractParameter,
        loc: Union[AbstractParameter, float],
        scale: AbstractParameter,
        gamma: AbstractParameter,
        slab: Union[AbstractParameter, float] = None,
    ) -> None:
        super().__init__(id_)
        self.x = x
        self.loc = loc
        self.gobal_scale = scale
        self.local_scale = gamma
        self.slab = slab

    def _call(self, *args, **kwargs) -> Tensor:
        if self.slab is not None:
            local_scale = (
                self.slab.tensor**2
                * self.local_scale.tensor**2
                / (
                    self.slab.tensor**2
                    + self.gobal_scale.tensor**2 * self.local_scale.tensor**2
                )
            ).sqrt()

        else:
            local_scale = self.local_scale.tensor
        return Normal(self.loc, self.gobal_scale.tensor * local_scale).log_prob(
            self.x.tensor
        )

    @property
    def sample_shape(self) -> Size:
        return self.x.tensor.shape[:-1]

    def handle_model_changed(self, model, obj, index) -> None:
        pass

    @staticmethod
    def json_factory(id_: str, x, loc, global_scale, local_scale, slab=None):
        model = {
            'id': id_,
            'type': 'ScaleMixtureNormal',
            'x': x,
            'loc': loc,
            'global_scale': global_scale,
            'local_scale': local_scale,
        }
        if slab is not None:
            model['slab'] = slab
        return model

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_object(data['x'], dic)
        loc = (
            float(data['loc'])
            if isinstance(data['loc'], (int, float))
            else process_object(data['loc'], dic)
        )
        global_scale = process_object(data['global_scale'], dic)
        local_scale = process_object(data['local_scale'], dic)
        if 'slab' in data:
            slab = process_object(data['slab'], dic)
        else:
            slab = None
        return cls(id_, x, loc, global_scale, local_scale, slab)
