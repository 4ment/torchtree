import collections
import collections.abc
import inspect
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ..core.abstractparameter import AbstractParameter
from ..core.container import Container
from ..core.model import CallableModel
from ..core.utils import get_class, process_objects, register_class
from ..typing import ID, OrderedDictType


@register_class
class Module(CallableModel):
    r"""Wrapper class for torch.nn.Module.

    :param id_: ID of object.
    :type id_: str or None
    :param torch.nn.Module module: a torch.nn.Module object.
    :param parameters: OrderedDict of :class:`~torchtree.Parameter` keyed by their ID.
    :type parameters: OrderedDict[str,Parameter]
    """

    def __init__(
        self,
        id_: ID,
        module: nn.Module,
        parameters: OrderedDictType[str, AbstractParameter],
    ) -> None:
        super().__init__(id_)
        self._module = module
        self.x = Container(None, parameters.values())

    @property
    def module(self) -> nn.Module:
        r"""
        :return: a pytorch Module object.
        :rtype: torch.nn.Module
        """
        return self._module

    def _call(self, *args, **kwargs) -> Tensor:
        return self._module()

    @property
    def sample_shape(self) -> torch.Size:
        raise NotImplementedError

    def to(self, *args, **kwargs) -> None:
        self._module = self._module.to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self._module = self._module.cuda(device)

    def cpu(self) -> None:
        self._module = self._module.cpu()

    @classmethod
    def from_json(cls, data, dic):
        r"""Create a Module object.

        :param data: json representation of Module object.
        :param dic: dictionary containing additional objects that can be referenced
         in data.

        :return: a :class:`~torchtree.nn.module.Module` object.
        :rtype: Module
        """
        klass = get_class(data['module'])
        signature_params = list(inspect.signature(klass.__init__).parameters)
        params: OrderedDict[str, AbstractParameter] = collections.OrderedDict()

        data_dist = data['parameters']
        for arg in signature_params[1:]:
            if arg in data_dist:
                if isinstance(data_dist[arg], str):
                    params[arg] = dic[data_dist[arg]]
                elif isinstance(data_dist[arg], dict):
                    params[arg] = process_objects(data_dist[arg], dic)
                else:
                    params[arg] = data_dist[arg]
            else:
                # other parameters that are not torchtree.Parameters
                pass
        module = klass(*[p.tensor for p in params.values()])

        return cls(data['id'], module, params)
