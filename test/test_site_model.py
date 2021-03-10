import numpy as np
import pytest
import torch

from phylotorch.evolution.site_model import ConstantSiteModel, WeibullSiteModel, InvariantSiteModel
from phylotorch.core.model import Parameter


def test_constant():
    sitemodel = ConstantSiteModel('constant')
    assert sitemodel.rates()[0] == 1.0
    assert sitemodel.probabilities()[0] == 1.0


def test_weibull():
    key = 'shape'
    sitemodel = WeibullSiteModel('weibull', Parameter(key, torch.tensor([1.])), 4)
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    sitemodel.update(torch.tensor([0.1]))
    new_rates_expected = (4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819)
    np.testing.assert_allclose(sitemodel.rates(), new_rates_expected, rtol=1e-06)

    sitemodel.update({key: torch.tensor([1.])})
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    assert torch.sum(sitemodel.rates() * sitemodel.probabilities()).item() == pytest.approx(1.0, 1.0e-6)


def test_weibull_invariant0():
    key = 'shape'
    sitemodel = WeibullSiteModel('weibull', Parameter(key, torch.tensor([1.])), 4, Parameter('inv', torch.tensor([0.])))
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    sitemodel.update({key: torch.tensor([0.1])})
    new_rates_expected = (4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819)
    np.testing.assert_allclose(sitemodel.rates(), new_rates_expected, rtol=1e-06)

    sitemodel.update({key: torch.tensor([1.])})
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    assert torch.sum(sitemodel.rates() * sitemodel.probabilities()).item() == pytest.approx(1.0, 1.0e-6)


def test_invariant():
    prop_invariant = torch.tensor([0.2])
    site_model = InvariantSiteModel('pinv', Parameter('inv', prop_invariant))
    rates = site_model.rates()
    props = site_model.probabilities()
    assert rates.mul(props).sum() == torch.tensor(np.ones(1))
    assert torch.all(torch.cat((prop_invariant, torch.tensor([0.8]))).eq(props))


def test_weibull_json():
    dic = {}
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    new_rates_expected = (4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819)
    sitemodel = WeibullSiteModel.from_json({'id': 'weibull',
                                            'type': 'phylotorch.evolution.sitemodel.WeibullSiteModel',
                                            'categories': 4,
                                            'shape': {
                                                'id': 'shape',
                                                'type': 'phylotorch.core.model.Parameter',
                                                'tensor': [1.]
                                            }}, dic)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)
    dic['shape'].tensor = torch.tensor(np.array([0.1]))
    np.testing.assert_allclose(sitemodel.rates(), new_rates_expected, rtol=1e-06)