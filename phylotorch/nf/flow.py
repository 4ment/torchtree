from typing import Tuple, List, Union, Dict

import torch.nn
from torch import nn, Tensor, Size

from ..core.model import Parameter
from ..core.utils import process_object, process_objects
from ..distributions.distributions import Distribution, DistributionModel
from ..nn.modules import Module


class NormalizingFlow(DistributionModel):
    """
    Class for normalizing flows.

    :param id_: ID of object
    :param x: parameter or list of parameters
    :param base: base distribution
    :param modules: list of transformations
    """

    def __init__(self, id_: str, x: Union[Parameter, List[Parameter]], base: Distribution,
                 modules: List[Module]) -> None:
        DistributionModel.__init__(self, id_)
        self.x = x
        self.base = base
        self.modules = modules
        self.layers = nn.ModuleList([t.module for t in modules])
        self.sum_log_abs_det_jacobians = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_J = 0.0
        z = x
        for layer in self.layers:
            y = layer(z)
            log_det_J += layer.log_abs_det_jacobian(y, z)
            z = y
        return z, log_det_J

    def apply_flow(self, sample_shape: Size):
        if sample_shape == torch.Size([]):
            zz, self.sum_log_abs_det_jacobians = self.forward(self.base.x.tensor.unsqueeze(0))
            zz = zz.squeeze()
        else:
            zz, self.sum_log_abs_det_jacobians = self.forward(self.base.x.tensor)

        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = zz[..., offset:(offset + xx.shape[-1])]
                offset += xx.shape[-1]
        else:
            self.x.tensor = zz

    def sample(self, sample_shape=Size()) -> None:
        self.base.sample(sample_shape)
        self.apply_flow(sample_shape)

    def rsample(self, sample_shape=Size()) -> None:
        self.base.rsample(sample_shape)
        self.apply_flow(sample_shape)

    def log_prob(self, x: Union[List[Parameter], Parameter] = None) -> Tensor:
        return self.base() - self.sum_log_abs_det_jacobians

    def _call(self, *args, **kwargs) -> Tensor:
        return self.log_prob()

    @property
    def batch_shape(self) -> torch.Size:
        return self.base.batch_shape

    @property
    def sample_shape(self) -> torch.Size:
        return self.base.sample_shape

    def parameters(self) -> List[Parameter]:
        parameters = []
        for module in self.modules:
            parameters.extend(module.parameters())
        return parameters

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data: Dict[str, any], dic: Dict[str, any]) -> 'NormalizingFlow':
        id_ = data['id']
        x = process_objects(data['x'], dic)
        base = process_object(data['base'], dic)
        modules = process_objects(data['layers'], dic)

        return cls(id_, x, base, modules)
