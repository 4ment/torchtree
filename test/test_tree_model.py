import pytest
import torch

from torchtree.evolution.tree_model import ReparameterizedTimeTreeModel


def node_heights_general_transform(
    id_: str, tree_id: str, ratios: list, root_height: list
) -> dict:
    node_heights = {
        'id': id_,
        'type': 'torchtree.TransformedParameter',
        'transform': 'torchtree.evolution.tree_model.GeneralNodeHeightTransform',
        'parameters': {'tree': tree_id},
        'x': [
            {
                'id': 'ratios',
                'type': 'torchtree.Parameter',
                'tensor': ratios,
            },
            {
                'id': 'root_height',
                'type': 'torchtree.Parameter',
                'tensor': root_height,
            },
        ],
    }
    return node_heights


@pytest.mark.parametrize(
    "ratios,root_height", [([2.0 / 6.0, 6.0 / 12.0], [12]), ([0.8, 0.2], [100])]
)
def test_general_node_height_transform(ratios, root_height):
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            ratios,
            root_height,
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
            **{'ratios_id': 'ratios', 'root_height_id': 'root_height'}
        ),
        dic,
    )
    expected = torch.log(
        tree_model.node_heights[-2] * tree_model.node_heights[-1]
    ).item()
    assert tree_model().item() == pytest.approx(expected, 0.0001)


@pytest.mark.parametrize(
    "ratios,root_height", [([2.0 / 6.0, 6.0 / 12.0], [12.0]), ([0.8, 0.2], [100.0])]
)
def test_general_node_height_transform_hetero(ratios, root_height):
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            ratios,
            root_height,
            dict(zip('ABCD', [0.0, 1.0, 4.0, 5.0])),
            **{'ratios_id': 'ratios', 'root_height_id': 'root_height'}
        ),
        dic,
    )
    expected = torch.log(
        (tree_model.node_heights[-2] - 1.0) * (tree_model.node_heights[-1] - 4.0)
    ).item()
    assert tree_model().item() == pytest.approx(expected, 0.0001)


@pytest.mark.parametrize(
    "ratios,root_height,keep,expected_ratios_root",
    [
        ([1.0 / 3.5, 1.5 / 4.0], [7.0], False, [1.0 / 3.5, 1.5 / 4.0, 7.0]),
        ([0.8, 0.2], [100.0], True, [1.0 / 3.5, 1.5 / 4.0, 7.0]),
    ],
)
def test_general_node_height_transform_hetero_all(
    ratios, root_height, keep, expected_ratios_root
):
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(A:2,(B:1.5,(C:2,D:1):2.5):2.5);',
            ratios,
            root_height,
            dict(zip('ABCD', [5.0, 3.0, 0.0, 1.0])),
            **{
                'keep_branch_lengths': keep,
                'ratios_id': 'ratios',
                'root_height_id': 'root_height',
            }
        ),
        dic,
    )
    expected = torch.tensor([5.0, 3.0, 0.0, 1.0, 2.0, 4.5, 7.0])
    expected_bounds = torch.tensor([5.0, 3.0, 0.0, 1.0, 1.0, 3.0, 5.0])
    expected_branch_lengths = torch.tensor([2.0, 1.5, 2.0, 1.0, 2.5, 2.5])
    log_det_jacobian = torch.log(expected[5] - expected_bounds[4]) + torch.log(
        expected[6] - expected_bounds[5]
    )
    assert torch.allclose(
        torch.tensor(expected_ratios_root),
        torch.cat((dic['ratios'].tensor, dic['root_height'].tensor)),
    )
    assert torch.allclose(expected, tree_model.node_heights)
    assert torch.allclose(expected_bounds, tree_model.transform._bounds)
    assert torch.allclose(expected_branch_lengths, tree_model.branch_lengths())
    assert torch.allclose(tree_model(), log_det_jacobian)


def test_general_node_height_transform_hetero_7():
    taxa = dict(zip('ABCDEFG', [5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0]))
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(A,(B,(C,(D,(E,(F,G))))));',
            [0.5] * (len(taxa) - 2),
            [10.0],
            taxa,
            **{'ratios_id': 'ratios', 'root_height_id': 'root_height'}
        ),
        dic,
    )
    log_det_jacobian = torch.tensor([0.0])
    for i in range(len(taxa), 2 * len(taxa) - 2):
        log_det_jacobian += (
            tree_model.node_heights[i + 1] - tree_model.transform._bounds[i]
        ).log()
    assert torch.allclose(tree_model(), log_det_jacobian)


@pytest.mark.parametrize(
    "ratios,root_height", [([2.0 / 6.0, 6.0 / 12.0], [12]), ([0.8, 0.2], [100])]
)
def test_general_node_height_heights_to_ratios(ratios, root_height):
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            ratios,
            root_height,
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
            **{'ratios_id': 'ratios', 'root_height_id': 'root_height'}
        ),
        {},
    )
    ratios_heights = tree_model.transform.inv(tree_model.node_heights[4:])
    assert torch.allclose(
        ratios_heights,
        torch.tensor(ratios + root_height, dtype=ratios_heights.dtype),
    )


def test_keep_branch_lengths_heights():
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '((((A_0:1.5,B_1:0.5):2.5,C_2:2):2,D_3:3):10,E_12:4);',
            [0.0] * 3,
            [0.0],
            dict(zip(['A_0', 'B_1', 'C_2', 'D_3', 'E_12'], [0.0, 1.0, 2.0, 3.0, 12.0])),
            **{
                'keep_branch_lengths': True,
                'ratios_id': 'ratios',
                'root_height_id': 'root_height',
            }
        ),
        dic,
    )
    assert torch.allclose(
        tree_model.branch_lengths(),
        torch.tensor([1.5, 0.5, 2.0, 3.0, 4.0, 2.5, 2.0, 10.0]),
    )
