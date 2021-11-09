import pytest
import torch

from torchtree.evolution.bdsk import PiecewiseConstantBirthDeath


def epidemio_to_bd(R, delta, s):
    r"""Convert epidemiology to birth death parameters

    :param R: effective reproductive number
    :param delta: total rate of becoming non infectious
    :param s: probability of an individual being sampled
    :return: lambda, mu, psi
    """
    lambda_ = R * delta
    mu = delta - s * delta
    psi = s * delta
    return lambda_, mu, psi


def test_1rho():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5])
    delta = torch.tensor([1.5])
    s = torch.zeros(1)

    lambda_, mu, psi = epidemio_to_bd(R, delta, s)
    rho = torch.tensor([0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, rho, origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -8.520565 == pytest.approx(log_p.item(), 0.0001)


def test_1rho2times():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0])
    delta = torch.tensor([1.5, 2.0])
    s = torch.zeros(2)

    lambda_, mu, psi = epidemio_to_bd(R, delta, s)
    rho = torch.tensor([0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, rho, origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -78.4006528776 == pytest.approx(log_p.item(), 0.0001)


def test_1rho2times_grid():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0])
    delta = torch.tensor([1.5, 2.0])
    s = torch.zeros(2)

    lambda_, mu, psi = epidemio_to_bd(R, delta, s)
    rho = torch.tensor([0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, rho, origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -78.4006528776 == pytest.approx(log_p.item(), 0.0001)


def test_1rho3times():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0, 5.0])
    delta = torch.tensor([1.5, 2.0, 1.0])
    s = torch.zeros(3)

    lambda_, mu, psi = epidemio_to_bd(R, delta, s)
    rho = torch.tensor([0.0, 0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, rho, origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -67.780094538296 == pytest.approx(log_p.item(), 0.0001)


def test_serial_1rho():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([2.0])
    mu = torch.tensor([1.0])
    psi = torch.tensor([0.5])
    rho = torch.tensor([0.0])

    origin = torch.tensor([6.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, rho, origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -19.0198 == pytest.approx(log_p.item(), 0.0001)


def test_serial_2rho():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([3.0, 2.0])
    mu = torch.tensor([2.5, 1.0])
    psi = torch.tensor([2.0, 0.5])
    rho = torch.zeros(2)

    origin = torch.tensor([6.0])
    times = torch.cat((torch.tensor([0.0, 3.0]), origin))

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho, origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -33.7573 == pytest.approx(log_p.item(), 0.0001)


def test_serial_3rho():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([3.0, 2.0, 4.0])
    mu = torch.tensor([2.5, 1.0, 0.5])
    psi = torch.tensor([2.0, 0.5, 1.0])
    rho = torch.zeros(3)

    origin = torch.tensor([6.0])
    times = torch.cat((torch.tensor([0.0, 3.0, 4.5]), origin))

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho, origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -37.8056 == pytest.approx(log_p.item(), 0.0001)


def test_serial_3rho_batch():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([[3.0, 2.0, 4.0], [13.0, 12.0, 14.0]])
    mu = torch.tensor([[2.5, 1.0, 0.5], [12.5, 11.0, 10.5]])
    psi = torch.tensor([[2.0, 0.5, 1.0], [2.0, 0.5, 1.0]])
    rho = torch.zeros((2, 3))

    origin = torch.tensor([6.0])
    times = torch.cat((torch.tensor([0.0, 3.0, 4.5]), origin))

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho, origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)).repeat(2, 1))
    assert torch.allclose(log_p, torch.tensor([-37.8056, -75.5726318359375]))
