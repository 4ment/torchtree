import torch
import torch.distributions

from phylotorch.core.model import CallableModel, Parameter
from phylotorch.core.utils import process_object


class PoissonTreeLikelihood(CallableModel):

    def __init__(self, id_, tree_model, clock_model, edge_lengths):
        super(PoissonTreeLikelihood, self).__init__(id_)
        self.tree_model = tree_model
        self.clock_model = clock_model
        self.edge_lengths = edge_lengths
        self.add_model(tree_model)
        self.add_model(clock_model)

    def __call__(self, *args, **kwargs):
        distances = self.tree_model.branch_lengths() * self.clock_model.rates
        return torch.distributions.Poisson(distances).log_prob(self.edge_lengths.tensor).sum()

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data['tree'], dic)
        clock_model = process_object(data['clockmodel'], dic)
        if 'edge_lengths' in data:
            if isinstance(data['edge_lengths'], list):
                edge_lengths = Parameter(None, torch.tensor(data['edge_lengths']))
            else:
                edge_lengths = process_object(data['edge_lengths'], dic)
        else:
            # use tree branch lengths
            edge_lengths = Parameter(None, tree_model.branch_lengths().detach().clone())
        if 'length' in data:
            edge_lengths.tensor = edge_lengths.tensor * data['length']
        return cls(id_, tree_model, clock_model, edge_lengths)
