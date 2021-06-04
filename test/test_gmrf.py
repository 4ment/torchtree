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
    assert torch.allclose(gmrf(), torch.tensor([[-1664.2894596388803], [-9.180924185988092]]))


@pytest.mark.parametrize("thetas,precision", [([2., 30., 4., 15., 6.], 20.0),
                                              ([1.0, 3.0, 6.0, 80.0, 9.0], 0.1)])
def test_gmrf2(thetas, precision):
    thetas = torch.tensor(thetas, requires_grad=True)
    thetas2 = thetas.detach()
    precision = torch.tensor([precision])
    gmrf = GMRF(None, x=Parameter(None, thetas),
                precision=Parameter(None, precision), tree_model=None)
    lp1 = gmrf()
    lp1.backward()

    thetas2.requires_grad = True
    dim = thetas.shape[0]
    Q = torch.zeros((dim, dim))
    Q[range(dim - 1), range(1, dim)] = -1
    Q[range(1, dim), range(dim - 1)] = -1
    Q.fill_diagonal_(2)
    Q[0, 0] = Q[dim - 1, dim - 1] = 1

    Q_scaled = Q * precision

    lp2 = 0.5 * (dim - 1) * precision.log() - 0.5 * torch.dot(thetas2, Q_scaled @ thetas2) - (
                dim - 1) / 2.0 * 1.8378770664093453
    lp2.backward()
    assert lp1.item() == pytest.approx(lp2.item())
    assert torch.allclose(thetas.grad, thetas2.grad)
