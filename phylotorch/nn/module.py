import collections
import collections.abc
import inspect
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ..core.model import CallableModel, Parameter
from ..core.utils import get_class, process_objects
from ..typing import ID, OrderedDictType


class Module(CallableModel):
    r"""Wrapper class for torch.nn.Module.

    :param id_: ID of object.
    :type id_: str or None
    :param torch.nn.Module module: a torch.nn.Module object.
    :param parameters: OrderedDict of :class:`~phylotorch.Parameter` keyed by their ID.
    :type parameters: OrderedDict[str,Parameter]
    """

    def __init__(
        self, id_: ID, module: nn.Module, parameters: OrderedDictType[str, Parameter]
    ) -> None:
        super().__init__(id_)
        self._module = module
        for parameter in parameters.values():
            self.add_parameter(parameter)

    @property
    def module(self) -> nn.Module:
        r"""
        :return: a pytorch Module object.
        :rtype: torch.nn.Module
        """
        return self._module

    def handle_model_changed(self, model, obj, index) -> None:
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event) -> None:
        self.fire_model_changed()

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

        :return: a :class:`~phylotorch.nn.module.Module` object.
        :rtype: Module
        """
        klass = get_class(data['module'])
        signature_params = list(inspect.signature(klass.__init__).parameters)
        params: OrderedDict[str, Parameter] = collections.OrderedDict()

        data_dist = data['parameters']
        for arg in signature_params[1:]:
            if arg in data_dist:
                if isinstance(data_dist[arg], str):
                    params[arg] = dic[data_dist[arg]]
                else:
                    params[arg] = process_objects(data_dist[arg], dic)
            else:
                # other parameters that are not phylotorch.Parameters
                pass
        module = klass(*[p.tensor for p in params.values()])

        return cls(data['id'], module, params)
