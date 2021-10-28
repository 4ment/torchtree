import pytest
import torch

from torchtree.evolution.tree_model import TimeTreeModel
from torchtree.evolution.tree_model_flexible import FlexibleTimeTreeModel


def node_heights_difference_transform(
    id_: str, tree_id: str, differences: list
) -> dict:
    node_heights = {
        'id': id_,
        'type': 'torchtree.TransformedParameter',
        'transform': 'torchtree.evolution.tree_height_transform.'
        'DifferenceNodeHeightTransform',
        'parameters': {'tree_model': tree_id},
        'x': {
            'id': 'differences',
            'type': 'torchtree.Parameter',
            'tensor': differences,
        },
    }
    return node_heights


@pytest.mark.parametrize(
    "differences",
    [torch.tensor([2.0] * 6), torch.arange(1.0, 7.0)],
)
def test_difference_height_transform_hetero(differences):
    taxa = dict(zip('ABCDEFG', [5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0]))
    node_heights = node_heights_difference_transform(
        'internal_heights', 'tree', differences.tolist()
    )

    dic = {}
    tree_model = FlexibleTimeTreeModel.from_json(
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
        torch.autograd.functional.jacobian(internal_heights.transform, differences)
        .det()
        .log(),
        internal_heights(),
    )
