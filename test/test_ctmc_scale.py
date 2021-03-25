import pytest
import torch

from phylotorch.core.model import Parameter
from phylotorch.distributions.ctmc_scale import CTMCScale
from phylotorch.evolution.tree_model import TimeTreeModel


def test_ctmc_scale():
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
        'newick': '((((A_0:1.5,B_1:0.5):2.5,C_2:2):2,D_3:3):10,E_12:4);',
        'node_heights': {
            "id": "a",
            "type": "phylotorch.core.model.Parameter",
            "tensor": [1.5, 4., 6., 16.]
        },
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A_0", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "B_1", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1.0}},
                {"id": "C_2", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 2.0}},
                {"id": "D_3", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 3.0}},
                {"id": "E_12", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 12.0}}
            ]
        }
    }
    tree_model = TimeTreeModel.from_json(tree_model, {})
    ctmc_scale = CTMCScale(None, Parameter(None, torch.tensor([0.001])), tree_model)
    assert 4.475351922659342 == pytest.approx(ctmc_scale().item(), 0.00001)
