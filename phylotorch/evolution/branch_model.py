from abc import abstractmethod

import torch

from ..core.model import Model, Parameter
from ..core.utils import process_object
from ..typing import ID
from .tree_model import TreeModel


class BranchModel(Model):
    _tag = 'branch_model'

    @property
    @abstractmethod
    def rates(self):
        pass


class AbstractClockModel(BranchModel):
    def __init__(self, id_: ID, rates: Parameter, tree: TreeModel) -> None:
        super().__init__(id_)
        self._rates = rates
        self.tree = tree
        self.add_parameter(rates)

    def update(self, value):
        if isinstance(value, dict):
            if self._rates.id in value:
                self._rates.set_tensor(value[self._rates.id])
        else:
            self._rates = value

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return self._rates.shape[:-1]


class StrictClockModel(AbstractClockModel):
    def __init__(self, id_: ID, rates: Parameter, tree: TreeModel) -> None:
        self.branch_count = tree.taxa_count * 2 - 2
        super().__init__(id_, rates, tree)

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


class SimpleClockModel(AbstractClockModel):
    @property
    def rates(self) -> torch.Tensor:
        return self._rates.tensor

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        rate = process_object(data['rate'], dic)
        return cls(id_, rate, tree_model)
