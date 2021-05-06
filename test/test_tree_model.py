import pytest
import torch

from phylotorch.evolution.tree_model import TimeTreeModel


def node_heights_general_transform(ratios, root_height):
    node_heights = {
        'id': 'node_heights',
        'type': 'phylotorch.core.model.TransformedParameter',
        'transform': 'phylotorch.evolution.tree_model.GeneralNodeHeightTransform',
        'parameters': {
            'tree': 'tree'
        },
        'x': [{
            'id': 'ratios',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': ratios
        },
            {
                'id': 'root_height',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': root_height
            }
        ]
    }
    return node_heights


def tree_model_transformed(newick, node_heights, taxa):
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
        'newick': newick,
        'node_heights': node_heights,
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [{"id": taxon, "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": date}} for
                     taxon, date in taxa.items()]
        }
    }
    return tree_model


@pytest.mark.parametrize("ratios,root_height", [([2. / 6., 6. / 12.], [12]),
                                                ([0.8, 0.2], [100])])
def test_GeneralNodeHeightTransform(ratios, root_height):
    taxa = {id_: date for id_, date in zip('ABCD', [0.0] * 4)}
    node_heights = node_heights_general_transform(ratios, root_height)
    tree_model = tree_model_transformed('(((A,B),C),D);', node_heights, taxa)
    dic = {}
    tree = TimeTreeModel.from_json(tree_model, dic)
    expected = torch.log(tree.node_heights[1] * tree.node_heights[2]).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)


@pytest.mark.parametrize("ratios,root_height", [([2. / 6., 6. / 12.], [12.]),
                                                ([0.8, 0.2], [100.])])
def test_GeneralNodeHeightTransform_hetero(ratios, root_height):
    taxa = {id_: date for id_, date in zip('ABCD', [0.0, 1.0, 4.0, 5.])}
    node_heights = node_heights_general_transform(ratios, root_height)
    tree_model = tree_model_transformed('(((A,B),C),D);', node_heights, taxa)
    dic = {}
    tree = TimeTreeModel.from_json(tree_model, dic)
    expected = torch.log((tree.node_heights[1] - 1.0) * (tree.node_heights[2] - 4.0)).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)


def test_GeneralNodeHeightTransform_hetero_all():
    taxa = {id_: date for id_, date in zip('ABCD', [5.0, 3.0, 0.0, 1.0])}
    node_heights = node_heights_general_transform([1. / 3.5, 1.5 / 4.], [7.])
    tree_model = tree_model_transformed('(A,(B,(C,D)));', node_heights, taxa)
    dic = {}
    tree = TimeTreeModel.from_json(tree_model, dic)
    expected = torch.tensor([2., 4.5, 7.], dtype=torch.float64)
    expected_bounds = torch.tensor([5., 3., 0., 1., 1., 3., 5.], dtype=torch.float64)
    expected_branch_lengths = torch.tensor([2., 1.5, 2., 1., 2.5, 2.5], dtype=torch.float64)
    log_det_jacobian = torch.log(expected[1] - expected_bounds[4]) + torch.log(expected[2] - expected_bounds[5])

    assert torch.allclose(expected, tree.node_heights)
    assert torch.allclose(expected_bounds, tree.bounds)
    assert torch.allclose(expected_branch_lengths, tree.branch_lengths())
    assert torch.allclose(dic['node_heights'](), log_det_jacobian)


def test_GeneralNodeHeightTransform_hetero_7():
    taxa = {id_: date for id_, date in zip('ABCDEFG', [5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0])}
    node_heights = node_heights_general_transform([0.5] * (len(taxa) - 2), [10.])
    tree_model = tree_model_transformed('(A,(B,(C,(D,(E,(F,G))))));', node_heights, taxa)
    dic = {}
    tree = TimeTreeModel.from_json(tree_model, dic)
    log_det_jacobian = torch.tensor([0], dtype=torch.float64)
    for i in range(len(taxa) - 2):
        log_det_jacobian += (tree.node_heights[i + 1] - tree.bounds[i + len(taxa)]).log()
    assert torch.allclose(dic['node_heights'](), log_det_jacobian)
