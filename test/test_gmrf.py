import pytest
import torch

from phylotorch.core.model import Parameter
from phylotorch.distributions.gmrf import GMRF


@pytest.mark.parametrize("thetas,precision,expected", [([2., 30., 4., 15., 6.], 2.0, -1664.2894596388803),
                                                       ([1.0, 3.0, 6.0, 8.0, 9.0], 0.1, -9.180924185988092)])
def test_gmrf(thetas, precision, expected):
    gmrf = GMRF(None, x=Parameter(None, torch.tensor(thetas)),
                precision=Parameter(None, torch.tensor([precision])), tree_model=None)
    assert expected == pytest.approx(gmrf().item(), 0.000001)


def test_gmrf_batch():
    gmrf = GMRF(None, x=Parameter(None, torch.tensor([[2., 30., 4., 15., 6.],
                                                      [1.0, 3.0, 6.0, 8.0, 9.0]])),
                precision=Parameter(None, torch.tensor([[2.], [0.1]])), tree_model=None)
    torch.allclose(gmrf(), torch.tensor([[-1664.2894596388803], [-9.180924185988092]]))
