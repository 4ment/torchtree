import pytest

from phylotorch.distributions.distributions import Distribution


def test_normal():
    normal = {
        'id': 'normal',
        'type': 'phylotorch.distributions.distributions.Distribution',
        'distribution': 'torch.distributions.normal.Normal',
        'parameters': {'loc': [1.0], 'scale': [2.0]},
        'x': {'id': 'x', 'type': 'phylotorch.core.model.Parameter', 'tensor': [3.0]},
    }
    distr = Distribution.from_json(normal, {})
    assert -2.112086 == pytest.approx(distr().item(), 0.0001)
