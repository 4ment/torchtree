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


@pytest.mark.parametrize("ages", ([5.0, 3.0, 0.0, 1.0, 0.0, 5.0, 6.0], [0.0] * 7))
@pytest.mark.parametrize(
    "differences",
    (torch.full((6,), 0.0), torch.arange(1.0, 7.0)),
)
@pytest.mark.parametrize(
    "newick",
    ('(A,(B,(C,(D,(E,(F,G))))));', '(A,((B,C),D),(E,(F,G)));'),
)
def test_difference_height_transform(ages, differences, newick):
    taxa = dict(zip('ABCDEFG', ages))
    node_heights = node_heights_difference_transform(
        'internal_heights', 'tree', differences.tolist()
    )

    dic = {}
    tree_model = FlexibleTimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            newick,
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
