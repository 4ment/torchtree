import numpy as np
import pytest
import torch

from torchtree import Parameter
from torchtree.evolution.site_model import (
    ConstantSiteModel,
    InvariantSiteModel,
    WeibullSiteModel,
)


@pytest.mark.parametrize(
    "mu,expected",
    (
        (None, torch.tensor([1.0])),
        (Parameter('mu', torch.tensor([2.0])), torch.tensor([2.0])),
    ),
)
def test_constant(mu, expected):
    sitemodel = ConstantSiteModel('constant', mu)
    assert torch.all(sitemodel.rates() == expected)
    assert sitemodel.probabilities()[0] == 1.0


def test_weibull_batch():
    key = 'shape'
    sitemodel = WeibullSiteModel(
        'weibull', Parameter(key, torch.tensor([[1.0], [0.1]])), 4
    )
    rates_expected = torch.tensor(
        [
            [0.1457844, 0.5131316, 1.0708310, 2.2702530],
            [4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819],
        ]
    )
    assert torch.allclose(sitemodel.rates(), rates_expected)


def test_weibull_batch2():
    key = 'shape'
    sitemodel = WeibullSiteModel(
        'weibull', Parameter(key, torch.tensor([[1.0], [0.1], [1.0]])), 4
    )
    rates_expected = torch.tensor(
        [
            [0.1457844, 0.5131316, 1.0708310, 2.2702530],
            [4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819],
            [0.1457844, 0.5131316, 1.0708310, 2.2702530],
        ]
    )
    assert torch.allclose(sitemodel.rates(), rates_expected)


def test_weibull():
    key = 'shape'
    sitemodel = WeibullSiteModel('weibull', Parameter(key, torch.tensor([1.0])), 4)
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    assert torch.sum(
        sitemodel.rates() * sitemodel.probabilities()
    ).item() == pytest.approx(1.0, 1.0e-6)


def test_weibull_invariant0():
    key = 'shape'
    sitemodel = WeibullSiteModel(
        'weibull',
        Parameter(key, torch.tensor([1.0])),
        4,
        Parameter('inv', torch.tensor([0.0])),
    )
    rates_expected = (0.0, 0.20506860315799713, 0.7796264290809631, 2.0153050422668457)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    assert torch.sum(
        sitemodel.rates() * sitemodel.probabilities()
    ).item() == pytest.approx(1.0, 1.0e-6)


def test_invariant_batch():
    prop_invariant = torch.tensor([[0.2], [0.3]])
    site_model = InvariantSiteModel('pinv', Parameter('inv', prop_invariant))
    rates = site_model.rates()
    props = site_model.probabilities()
    assert torch.all(rates.mul(props).sum(-1) == torch.ones(2))


def test_invariant():
    prop_invariant = torch.tensor([0.2])
    site_model = InvariantSiteModel('pinv', Parameter('inv', prop_invariant))
    rates = site_model.rates()
    props = site_model.probabilities()
    assert rates.mul(props).sum() == torch.tensor(np.ones(1))
    assert torch.all(torch.cat((prop_invariant, torch.tensor([0.8]))).eq(props))


def test_invariant_mu():
    prop_invariant = torch.tensor([0.2])
    site_model = InvariantSiteModel(
        'pinv', Parameter('inv', prop_invariant), Parameter('mu', torch.tensor([2.0]))
    )
    assert site_model.rates()[1] == 2 / 0.8


def test_weibull_json():
    dic = {}
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    new_rates_expected = (4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819)
    sitemodel = WeibullSiteModel.from_json(
        {
            'id': 'weibull',
            'type': 'torchtree.evolution.sitemodel.WeibullSiteModel',
            'categories': 4,
            'shape': {
                'id': 'shape',
                'type': 'torchtree.Parameter',
                'tensor': [1.0],
            },
        },
        dic,
    )
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)
    dic['shape'].tensor = torch.tensor(np.array([0.1]))
    np.testing.assert_allclose(sitemodel.rates(), new_rates_expected, rtol=1e-06)


def test_weibull_mu():
    sitemodel = WeibullSiteModel(
        'weibull',
        Parameter('shape', torch.tensor([1.0])),
        4,
        mu=Parameter('mu', torch.tensor([2.0])),
    )
    rates_expected = torch.tensor([0.1457844, 0.5131316, 1.0708310, 2.2702530]) * 2
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)
