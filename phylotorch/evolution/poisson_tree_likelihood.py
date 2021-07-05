import torch
import torch.distributions

from phylotorch.core.model import CallableModel, Parameter
from phylotorch.core.utils import process_object
from phylotorch.evolution.branch_model import BranchModel
from phylotorch.evolution.tree_model import TreeModel, TimeTreeModel
from phylotorch.typing import ID


class PoissonTreeLikelihood(CallableModel):
    r"""
    Tree likelihood class using Poisson model.

    :param id_: ID of object.
    :type id_: str or None
    :param TimeTreeModel tree_model: a tree model.
    :param BranchModel clock_model: a clock model.
    :param Parameter edge_lengths: edge lengths.
    """

    def __init__(self, id_: ID, tree_model: TimeTreeModel, clock_model: BranchModel, edge_lengths: Parameter) -> None:
        super(PoissonTreeLikelihood, self).__init__(id_)
        self.tree_model = tree_model
        self.clock_model = clock_model
        self.edge_lengths = edge_lengths
        self.add_model(tree_model)
        self.add_model(clock_model)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        distances = self.tree_model.branch_lengths() * self.clock_model.rates
        return torch.distributions.Poisson(distances).log_prob(self.edge_lengths.tensor).sum()

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic) -> 'PoissonTreeLikelihood':
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        clock_model = process_object(data[BranchModel.tag], dic)
        if 'edge_lengths' in data:
            if isinstance(data['edge_lengths'], list):
                edge_lengths = Parameter(None, torch.tensor(data['edge_lengths'], dtype=torch.long))
            else:
                edge_lengths = process_object(data['edge_lengths'], dic)
        else:
            # use tree branch lengths
            edge_lengths = Parameter(None, tree_model.branch_lengths().detach().clone())
        if 'length' in data:
            edge_lengths.tensor = edge_lengths.tensor * data['length']
        return cls(id_, tree_model, clock_model, edge_lengths)
