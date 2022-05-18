from __future__ import annotations

from typing import Optional, Union

import torch.nn
from torch import Size, Tensor, nn

from ..core.abstractparameter import AbstractParameter
from ..core.utils import get_class, process_object, process_objects, register_class
from ..distributions.distributions import Distribution, DistributionModel
from ..nn.module import Module


@register_class
class NormalizingFlow(DistributionModel):
    r"""
    Class for normalizing flows.

    :param id_: ID of object
    :type id_: str or None
    :param x: parameter or list of parameters
    :type x: List[Parameter]
    :param Distribution base: base distribution
    :param modules: list of transformations
    :type modules: List[Module]
    """

    def __init__(
        self,
        id_: str,
        x: Union[AbstractParameter, list[AbstractParameter]],
        base: Distribution,
        modules: list[Module],
        dtype=None,
        device=None,
    ) -> None:
        DistributionModel.__init__(self, id_)
        self.x = x
        self.base = base
        self.modules = modules
        self.layers = nn.ModuleList([t.module for t in modules])
        self.sum_log_abs_det_jacobians = None
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_det_J = 0.0
        z = x
        for layer in self.layers:
            y = layer(z)
            log_det_J += layer.log_abs_det_jacobian(y, z)
            z = y
        return z, log_det_J

    def apply_flow(self, sample_shape: Size):
        if sample_shape == torch.Size([]):
            zz, self.sum_log_abs_det_jacobians = self.forward(
                self.base.x.tensor.unsqueeze(0)
            )
            zz = zz.squeeze()
        else:
            zz, self.sum_log_abs_det_jacobians = self.forward(self.base.x.tensor)

        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = zz[..., offset : (offset + xx.shape[-1])]
                offset += xx.shape[-1]
        else:
            self.x.tensor = zz

    def sample(self, sample_shape=Size()) -> None:
        self.base.sample(sample_shape)
        self.apply_flow(sample_shape)

    def rsample(self, sample_shape=Size()) -> None:
        self.base.rsample(sample_shape)
        self.apply_flow(sample_shape)

    def log_prob(
        self, x: Union[list[AbstractParameter], AbstractParameter] = None
    ) -> Tensor:
        return self.base() - self.sum_log_abs_det_jacobians

    def entropy(self) -> torch.Tensor:
        raise RuntimeError('Cannot use entropy with NormalizingFlow class')

    def _call(self, *args, **kwargs) -> Tensor:
        return self.log_prob()

    @property
    def sample_shape(self) -> torch.Size:
        return self.base.sample_shape

    def parameters(self) -> list[AbstractParameter]:
        parameters = []
        for module in self.modules:
            parameters.extend(module.parameters())
        return parameters

    def to(self, *args, **kwargs) -> None:
        for module in self.modules:
            module.to(*args, **kwargs)
        self.base.to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        for module in self.modules:
            module.cuda(device)
        self.base.cuda(device)

    def cpu(self) -> None:
        for module in self.modules:
            module.cpu()
        self.base.cpu()

    @classmethod
    def from_json(cls, data: dict[str, any], dic: dict[str, any]) -> NormalizingFlow:
        r"""
        Create a Flow object.

        :param data: json representation of Flow object.
        :param dic: dictionary containing additional objects that can be
         referenced in data.

        :return: a :class:`~torchtree.nn.flow.NormalizingFlow` object.
        :rtype: NormalizingFlow
        """
        id_ = data['id']
        x = process_objects(data['x'], dic)
        base = process_object(data['base'], dic)
        modules = process_objects(data['layers'], dic)
        if 'dtype' in data:
            dtype = get_class(data['dtype'])
        else:
            dtype = None
        device = data.get('device', None)

        return cls(id_, x, base, modules, dtype, device)
