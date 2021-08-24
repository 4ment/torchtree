import torch
import torch.distributions

from phylotorch.evolution.poisson_tree_likelihood import PoissonTreeLikelihood
from phylotorch.evolution.tree_model import TimeTreeModel


def test_poisson_json():
    dic = {}
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
            **{'internal_heights': [10.0, 20.0, 30.0]}
        ),
        dic,
    )

    distances = tree_model.branch_lengths()
    noisy_distances = torch.clamp(distances * torch.rand(1), min=0.0).long()

    poisson_model = {
        'id': 'a',
        'type': 'phylotorch.evolution.poisson_tree_likelihood.PoissonTreeLikelihood',
        'tree_model': 'tree',
        'edge_lengths': noisy_distances.tolist(),
        'branch_model': {
            'id': 'clock',
            'type': 'phylotorch.evolution.branch_model.StrictClockModel',
            'tree_model': 'tree',
            'rate': {
                'id': 'rate',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': [0.01],
            },
        },
    }

    poisson = PoissonTreeLikelihood.from_json(poisson_model, dic)
    assert torch.allclose(
        poisson(),
        torch.distributions.Poisson(distances * 0.01).log_prob(noisy_distances).sum(),
    )
