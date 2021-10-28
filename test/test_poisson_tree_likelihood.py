import torch
import torch.distributions

from torchtree.evolution.poisson_tree_likelihood import PoissonTreeLikelihood
from torchtree.evolution.tree_model import ReparameterizedTimeTreeModel


def test_poisson_json():
    dic = {}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            [10 / 20, 20 / 30],
            [30.0],
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
        ),
        dic,
    )
    dic['tree'] = tree_model

    distances = tree_model.branch_lengths()
    noisy_distances = torch.clamp(distances * torch.rand(1), min=0.0).long()

    poisson_model = {
        'id': 'a',
        'type': 'torchtree.evolution.poisson_tree_likelihood.PoissonTreeLikelihood',
        'tree_model': 'tree',
        'edge_lengths': noisy_distances.tolist(),
        'branch_model': {
            'id': 'clock',
            'type': 'torchtree.evolution.branch_model.StrictClockModel',
            'tree_model': 'tree',
            'rate': {
                'id': 'rate',
                'type': 'torchtree.Parameter',
                'tensor': [0.01],
            },
        },
    }

    poisson = PoissonTreeLikelihood.from_json(poisson_model, dic)
    assert torch.allclose(
        poisson(),
        torch.distributions.Poisson(distances * 0.01).log_prob(noisy_distances).sum(),
    )
