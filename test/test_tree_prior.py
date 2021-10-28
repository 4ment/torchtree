import torch

from torchtree import Parameter
from torchtree.distributions.tree_prior import CompoundGammaDirichletPrior
from torchtree.evolution.tree_model import UnRootedTreeModel


def test_prior_mrbayes():
    dic = {}
    json_tree = UnRootedTreeModel.json_factory(
        'tree',
        '(6:0.02,((5:0.02,2:0.02):0.02,(4:0.02,3:0.02):0.02):0.02,1:0.02);',
        [0.0],
        {str(i): None for i in range(1, 7)},
        **{
            'keep_branch_lengths': True,
            'branch_lengths_id': 'bl',
        }
    )

    tree_model = UnRootedTreeModel.from_json(json_tree, dic)
    prior = CompoundGammaDirichletPrior(
        None,
        tree_model,
        Parameter(None, torch.tensor([1.0])),
        Parameter(None, torch.tensor([1.0])),
        Parameter(None, torch.tensor([1.0])),
        Parameter(None, torch.tensor([0.1])),
    )
    assert torch.allclose(prior(), torch.tensor([22.00240516662597]))
