from abc import ABC, abstractmethod

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import Model
from ..core.utils import process_object, register_class
from ..typing import ID
from .tree_model import TreeModel


class BranchModel(Model):
    _tag = 'branch_model'

    @property
    @abstractmethod
    def rates(self):
        pass


class AbstractClockModel(BranchModel, ABC):
    def __init__(self, id_: ID, rates: AbstractParameter, tree: TreeModel) -> None:
        super().__init__(id_)
        self._rates = rates
        self.tree = tree

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return self._rates.shape[:-1]


@register_class
class StrictClockModel(AbstractClockModel):
    def __init__(self, id_: ID, rates: AbstractParameter, tree: TreeModel) -> None:
        super().__init__(id_, rates, tree)
        self.branch_count = tree.taxa_count * 2 - 2

    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor.expand(
            [-1] * (self._rates.tensor.dim() - 1) + [self.branch_count]
        )

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        rate = process_object(data['rate'], dic)
        return cls(id_, rate, tree_model)


@register_class
class SimpleClockModel(AbstractClockModel):
    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor

    @staticmethod
    def json_factory(id_: str, tree_model, rate):
        return {
            'id': id_,
            'type': 'SimpleClockModel',
            TreeModel.tag: tree_model,
            'rate': rate,
        }

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        rate = process_object(data['rate'], dic)
        return cls(id_, rate, tree_model)
