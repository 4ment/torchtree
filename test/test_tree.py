import pytest

import torch

from phylotorch.evolution.tree import GeneralNodeHeightTransform, TimeTreeModel


@pytest.fixture
def ratios_list():
    return 2. / 6., 6. / 12., 12.


@pytest.fixture
def node_heights_transformed(ratios_list):
    node_heights = {
            'id': 'node_heights',
            'type': 'phylotorch.core.model.TransformedParameter',
            'transform': {
                'id': 'nodeTransform',
                'type': 'phylotorch.core.model.TransformModel',
                'transform': 'phylotorch.evolution.tree.GeneralNodeHeightTransform',
                'parameters': {
                    'tree': 'tree'
                }
            },
            'x': [{
                    'id': 'ratios',
                    'type': 'phylotorch.core.model.Parameter',
                    'tensor': ratios_list[:-1]
                },
                {
                    'id': 'root_height',
                    'type': 'phylotorch.core.model.Parameter',
                    'tensor': ratios_list[-1:]
                }
            ]
        }
    return node_heights


@pytest.fixture
def tree_model_node_heights_transformed(node_heights_transformed):
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': node_heights_transformed,
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
    return tree_model

@pytest.fixture
def tree_model_node_heights_transformed_hetero(node_heights_transformed):
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': node_heights_transformed,
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "B", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1.0}},
                {"id": "C", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 4.0}},
                {"id": "D", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 5.0}}
            ]
        }
    }
    return tree_model


def test_GeneralNodeHeightTransform(tree_model_node_heights_transformed):
    dic = {}
    tree = TimeTreeModel.from_json(tree_model_node_heights_transformed, dic)
    expected = torch.log(tree.node_heights[1] * tree.node_heights[2]).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)
    assert dic['nodeTransform'].log_abs_det_jacobian(None, tree.node_heights).item() == pytest.approx(expected, 0.0001)
    print(tree.node_heights)


def test_GeneralNodeHeightTransform_hetero(tree_model_node_heights_transformed_hetero):
    dic = {}
    tree = TimeTreeModel.from_json(tree_model_node_heights_transformed_hetero, dic)
    expected = torch.log((tree.node_heights[1] - 1.0) * (tree.node_heights[2] - 4.0)).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)
    assert dic['nodeTransform'].log_abs_det_jacobian(None, tree.node_heights).item() == pytest.approx(expected, 0.0001)
