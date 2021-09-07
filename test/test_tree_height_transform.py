import pytest
import torch

from phylotorch.evolution.tree_model import TimeTreeModel


def node_heights_difference_transform(
    id_: str, tree_id: str, differences: list
) -> dict:
    node_heights = {
        'id': id_,
        'type': 'phylotorch.TransformedParameter',
        'transform': 'phylotorch.evolution.tree_height_transform.'
        'DifferenceNodeHeightTransform',
        'parameters': {'tree': tree_id},
        'x': {
            'id': 'differences',
            'type': 'phylotorch.Parameter',
            'tensor': differences,
        },
    }
    return node_heights


@pytest.mark.skip
def test_difference_height_transform_hetero():
    taxa = dict(zip('ABCDEFG', [5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0]))
    differences = torch.tensor([2.0] * (len(taxa) - 1))
    node_heights = node_heights_difference_transform(
        'internal_heights', 'tree', differences.tolist()
    )

    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(A,(B,(C,(D,(E,(F,G))))));',
            node_heights,
            taxa,
        ),
        dic,
    )
    internal_heights = dic['internal_heights']
    assert torch.allclose(internal_heights.x.tensor, differences)

    assert torch.allclose(
        internal_heights.transform.inv(
            tree_model.node_heights[tree_model.taxa_count :]
        ),
        differences,
    )

    assert torch.allclose(
        internal_heights(),
        differences.sum(0),
    )
