import pytest
import torch
import torch.distributions

from phylotorch.evolution.poisson_treelikelihood import PoissonTreeLikelihood
from phylotorch.evolution.tree import heights_to_branch_lengths, TimeTreeModel


def test_poisson_json():
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': {
            'id': 'node_heights',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [10.0, 20.0, 30.0]
        },
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "B", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "C", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "D", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}}
            ]
        }
    }

    dic = {}
    tree = TimeTreeModel.from_json(tree_model, dic)
    distances = heights_to_branch_lengths(tree.node_heights, tree.bounds, tree.preorder)
    noisy_distances = torch.clamp(distances * torch.rand(1), min=0.0)

    poisson_model = {
        'id': 'a',
        'type': 'phylotorch.evolution.poisson_treelikelihood.PoissonTreeLikelihood',
        'tree': 'tree',
        'edge_lengths': noisy_distances.tolist(),
        'clockmodel': {
            'id': 'clock',
            'type': 'phylotorch.evolution.clockmodel.StrictClockModel',
            'tree': 'tree',
            'rate': {
                'id': 'rate',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': [0.01]
            }
        }
    }

    poisson = PoissonTreeLikelihood.from_json(poisson_model, dic)
    assert pytest.approx(poisson().item(), 1e-5) == torch.distributions.Poisson(distances * 0.01).log_prob(
        noisy_distances).sum().item()
