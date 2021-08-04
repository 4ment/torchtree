import pytest
import torch

from phylotorch.evolution.tree_model import TimeTreeModel


def node_heights_general_transform(
    id_: str, tree_id: str, ratios: list, root_height: list
) -> dict:
    node_heights = {
        'id': id_,
        'type': 'phylotorch.core.model.TransformedParameter',
        'transform': 'phylotorch.evolution.tree_model.GeneralNodeHeightTransform',
        'parameters': {'tree': tree_id},
        'x': [
            {
                'id': 'ratios',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': ratios,
            },
            {
                'id': 'root_height',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': root_height,
            },
        ],
    }
    return node_heights


@pytest.mark.parametrize(
    "ratios,root_height", [([2.0 / 6.0, 6.0 / 12.0], [12]), ([0.8, 0.2], [100])]
)
def test_general_node_height_transform(ratios, root_height):
    node_heights = node_heights_general_transform(
        'node_heights', 'tree', ratios, root_height
    )
    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
            **{'node_heights': node_heights}
        ),
        dic,
    )
    expected = torch.log(tree_model.node_heights[1] * tree_model.node_heights[2]).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)


@pytest.mark.parametrize(
    "ratios,root_height", [([2.0 / 6.0, 6.0 / 12.0], [12.0]), ([0.8, 0.2], [100.0])]
)
def test_general_node_height_transform_hetero(ratios, root_height):
    node_heights = node_heights_general_transform(
        'node_heights', 'tree', ratios, root_height
    )
    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            dict(zip('ABCD', [0.0, 1.0, 4.0, 5.0])),
            **{'node_heights': node_heights}
        ),
        dic,
    )
    expected = torch.log(
        (tree_model.node_heights[1] - 1.0) * (tree_model.node_heights[2] - 4.0)
    ).item()
    assert dic['node_heights']().item() == pytest.approx(expected, 0.0001)


def test_general_node_height_transform_hetero_all():
    node_heights = node_heights_general_transform(
        'node_heights', 'tree', [1.0 / 3.5, 1.5 / 4.0], [7.0]
    )

    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(A,(B,(C,D)));',
            dict(zip('ABCD', [5.0, 3.0, 0.0, 1.0])),
            **{'node_heights': node_heights}
        ),
        dic,
    )
    expected = torch.tensor([2.0, 4.5, 7.0], dtype=torch.float64)
    expected_bounds = torch.tensor(
        [5.0, 3.0, 0.0, 1.0, 1.0, 3.0, 5.0], dtype=torch.float64
    )
    expected_branch_lengths = torch.tensor(
        [2.0, 1.5, 2.0, 1.0, 2.5, 2.5], dtype=torch.float64
    )
    log_det_jacobian = torch.log(expected[1] - expected_bounds[4]) + torch.log(
        expected[2] - expected_bounds[5]
    )

    assert torch.allclose(expected, tree_model.node_heights)
    assert torch.allclose(expected_bounds, tree_model.bounds)
    assert torch.allclose(expected_branch_lengths, tree_model.branch_lengths())
    assert torch.allclose(dic['node_heights'](), log_det_jacobian)


def test_general_node_height_transform_hetero_7():
    taxa = dict(zip('ABCDEFG', [5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0]))
    node_heights = node_heights_general_transform(
        'node_heights', 'tree', [0.5] * (len(taxa) - 2), [10.0]
    )

    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree', '(A,(B,(C,(D,(E,(F,G))))));', taxa, **{'node_heights': node_heights}
        ),
        dic,
    )
    log_det_jacobian = torch.tensor([0], dtype=torch.float64)
    for i in range(len(taxa) - 2):
        log_det_jacobian += (
            tree_model.node_heights[i + 1] - tree_model.bounds[i + len(taxa)]
        ).log()
    assert torch.allclose(dic['node_heights'](), log_det_jacobian)
