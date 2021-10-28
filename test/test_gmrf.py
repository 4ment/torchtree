import pytest
import torch

from torchtree import Parameter
from torchtree.distributions.gmrf import GMRF, GMRFCovariate
from torchtree.evolution.tree_model import TimeTreeModel


@pytest.mark.parametrize(
    "thetas,precision,expected",
    [
        ([2.0, 30.0, 4.0, 15.0, 6.0], 2.0, -1664.2894596388803),
        ([1.0, 3.0, 6.0, 8.0, 9.0], 0.1, -9.180924185988092),
    ],
)
def test_gmrf(thetas, precision, expected):
    gmrf = GMRF(
        None,
        field=Parameter(None, torch.tensor(thetas)),
        precision=Parameter(None, torch.tensor([precision])),
        tree_model=None,
    )
    assert expected == pytest.approx(gmrf().item(), 0.000001)


def test_gmrf_batch():
    gmrf = GMRF(
        None,
        field=Parameter(
            None, torch.tensor([[2.0, 30.0, 4.0, 15.0, 6.0], [1.0, 3.0, 6.0, 8.0, 9.0]])
        ),
        precision=Parameter(None, torch.tensor([[2.0], [0.1]])),
        tree_model=None,
    )
    assert torch.allclose(
        gmrf(), torch.tensor([[-1664.2894596388803], [-9.180924185988092]])
    )


def test_gmrfw2():
    gmrf = GMRF(
        None,
        field=Parameter(None, torch.tensor([3.0, 10.0, 4.0]).log()),
        precision=Parameter(None, torch.tensor([0.1])),
        tree_model=None,
    )
    assert -4.254919053937792 == pytest.approx(gmrf().item(), 0.000001)


@pytest.mark.parametrize(
    "rescale,expected",
    [
        (True, -4.501653235865269),
        (False, -4.230759878711851),
    ],
)
def test_smoothed(rescale, expected):
    tree_model = TimeTreeModel.from_json(
        TimeTreeModel.json_factory(
            'tree',
            '(((A,B),C),D);',
            [3.0, 2.0, 4.0],
            dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
        ),
        {},
    )

    gmrf = GMRF(
        None,
        field=Parameter(None, torch.tensor([3.0, 10.0, 4.0]).log()),
        precision=Parameter(None, torch.tensor([0.1])),
        tree_model=tree_model,
        rescale=rescale,
    )
    assert torch.allclose(torch.tensor(expected), gmrf())


@pytest.mark.parametrize(
    "thetas,precision",
    [([2.0, 30.0, 4.0, 15.0, 6.0], 20.0), ([1.0, 3.0, 6.0, 80.0, 9.0], 0.1)],
)
def test_gmrf2(thetas, precision):
    thetas = torch.tensor(thetas, requires_grad=True)
    thetas2 = thetas.detach()
    precision = torch.tensor([precision])
    gmrf = GMRF(
        None,
        field=Parameter(None, thetas),
        precision=Parameter(None, precision),
        tree_model=None,
    )
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

    lp2 = (
        0.5 * (dim - 1) * precision.log()
        - 0.5 * torch.dot(thetas2, Q_scaled @ thetas2)
        - (dim - 1) / 2.0 * 1.8378770664093453
    )
    lp2.backward()
    assert lp1.item() == pytest.approx(lp2.item())
    assert torch.allclose(thetas.grad, thetas2.grad)


@pytest.mark.parametrize(
    'thetas', [[2.0, 30.0, 4.0, 15.0, 6.0], [1.0, 3.0, 6.0, 80.0, 9.0]]
)
@pytest.mark.parametrize('precision', [20.0, 0.1])
@pytest.mark.parametrize('weights', [None, [0.12, 0.2, 0.28, 0.26]])
@pytest.mark.parametrize('rescale', [True])
def test_gmrf_time_aware(thetas, precision, weights, rescale):
    thetas = torch.tensor(thetas, requires_grad=True)

    precision = torch.tensor([precision])
    weights_tensor = torch.tensor(weights) if weights is not None else None
    gmrf = GMRF(
        None,
        field=Parameter(None, thetas),
        precision=Parameter(None, precision),
        tree_model=None,
        weights=weights_tensor,
        rescale=rescale,
    )
    lp1 = gmrf()

    dim = thetas.shape[0]
    if weights is not None:
        times = torch.tensor([0.0, 2.0, 6.0, 12.0, 20.0, 25.0])
        durations = times[..., 1:] - times[..., :-1]
        offdiag = -2.0 / (durations[..., :-1] + durations[..., 1:])
        if rescale:
            offdiag *= times[-1]  # rescale with root height
    else:
        offdiag = torch.full((dim - 1,), -1.0)

    Q = torch.zeros((dim, dim))
    Q[range(dim - 1), range(1, dim)] = offdiag
    Q[range(1, dim), range(dim - 1)] = offdiag

    Q[range(1, dim - 1), range(1, dim - 1)] = -(offdiag[..., :-1] + offdiag[..., 1:])
    Q[0, 0] = -Q[0, 1]
    Q[dim - 1, dim - 1] = -offdiag[-1]

    Q_scaled = Q * precision
    lp2 = (
        0.5 * (dim - 1) * precision.log()
        - 0.5 * torch.dot(thetas, Q_scaled @ thetas)
        - (dim - 1) / 2.0 * 1.8378770664093453
    )

    assert lp1.item() == pytest.approx(lp2.item())


def test_gmrf_covariates_simple():
    gmrf = GMRF(
        None,
        Parameter(None, torch.tensor([1.0, 2.0, 3.0])),
        Parameter(None, torch.tensor([0.1])),
    )
    gmrf_covariate = GMRFCovariate(
        None,
        field=Parameter(None, torch.tensor([1.0, 2.0, 3.0])),
        precision=Parameter(None, torch.tensor([0.1])),
        covariates=Parameter(None, torch.arange(1.0, 7.0).view((3, 2))),
        beta=Parameter(None, torch.tensor([0.0, 0.0])),
    )
    assert gmrf() == gmrf_covariate()
