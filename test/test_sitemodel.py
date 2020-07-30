import numpy as np
import pytest
import torch

from phylotorch.sitemodel import ConstantSiteModel, WeibullSiteModel


def test_constant():
    sitemodel = ConstantSiteModel()
    assert sitemodel.rates()[0] == 1.0
    assert sitemodel.probabilities()[0] == 1.0


def test_weibull():
    key = 'shape'
    sitemodel = WeibullSiteModel((key, torch.tensor([1.])), 4)
    rates_expected = (0.1457844, 0.5131316, 1.0708310, 2.2702530)
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    sitemodel.update(torch.tensor([0.1]))
    new_rates_expected = (4.766392e-12, 1.391131e-06, 2.179165e-03, 3.997819)
    np.testing.assert_allclose(sitemodel.rates(), new_rates_expected, rtol=1e-06)

    sitemodel.update({key: torch.tensor([1.])})
    np.testing.assert_allclose(sitemodel.rates(), rates_expected, rtol=1e-06)

    assert torch.sum(sitemodel.rates() * sitemodel.probabilities()).item() == pytest.approx(1.0, 1.0e-6)
