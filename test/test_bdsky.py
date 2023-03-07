import pytest
import torch

from torchtree.evolution.bdsk import (
    PiecewiseConstantBirthDeath,
    epidemiology_to_birth_death,
)

# Tests are adapted from
# https://github.com/BEAST2-Dev/bdsky/blob/master/test/beast/evolution/speciation/BirthDeathSkylineTest.java


def test_single_rho():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5])
    delta = torch.tensor([1.5])
    s = torch.zeros(1)

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)
    rho = torch.tensor([0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -8.520565 == pytest.approx(log_p.item(), 0.0001)

    origin = torch.tensor([1.0e-100])
    bdsky = PiecewiseConstantBirthDeath(
        lambda_,
        mu,
        psi,
        rho=rho,
        origin=origin,
        origin_is_root_edge=True,
        survival=False,
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -5.950979 == pytest.approx(log_p.item(), 0.0001)

    origin = torch.tensor([1.0e-100])
    bdsky = PiecewiseConstantBirthDeath(
        lambda_,
        mu,
        psi,
        rho=rho,
        origin=origin,
        origin_is_root_edge=True,
        survival=True,
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -4.431935 == pytest.approx(log_p.item(), 0.0001)

    origin = torch.tensor([10.0])
    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, survival=True
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -7.404227 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculationSimple
def testLikelihood_calculation_simple():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    R = torch.tensor([1.5])
    delta = torch.tensor([1.5])
    s = torch.tensor([0.3])

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)
    removal_probability = torch.ones(1)

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_,
        mu,
        psi,
        origin=origin,
        removal_probability=removal_probability,
        survival=False,
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -26.105360134266082 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculationSimpleForBDMM
def test_likelihood_calculation_simple_for_BDMM():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    R = torch.tensor([1.5])
    delta = torch.tensor([1.5])
    s = torch.tensor([0.3])
    removal_probability = torch.tensor([0.9])

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s, removal_probability)

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_,
        mu,
        psi,
        origin=origin,
        removal_probability=removal_probability,
        survival=True,
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -25.991511346557598 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculation1
def test_likelihood_calculation1():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([2.0])
    mu = torch.tensor([1.0])
    psi = torch.tensor([0.5])

    origin = torch.tensor([6.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, origin=origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -19.0198 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculation4
def test_likelihood_calculation4():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([3.0, 2.0])
    mu = torch.tensor([2.5, 1.0])
    psi = torch.tensor([2.0, 0.5])

    origin = torch.tensor([6.0])
    times = torch.tensor([0.0, 3.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, origin=origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -33.7573 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculation5
def test_likelihood_calculation5():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([3.0, 2.0, 4.0])
    mu = torch.tensor([2.5, 1.0, 0.5])
    psi = torch.tensor([2.0, 0.5, 1.0])

    origin = torch.tensor([6.0])
    times = torch.tensor([0.0, 3.0, 4.5])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, origin=origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -37.8056 == pytest.approx(log_p.item(), 0.0001)


# testLikelihoodCalculation6
def test_likelihood_calculation6():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    R = torch.tensor([2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0])
    delta = torch.tensor([4.5, 1.5, 1.5])
    s = torch.tensor([4.0 / 9.0, 1.0 / 3.0, 2.0 / 3.0])

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)

    origin = torch.tensor([6.0])
    times = torch.tensor([0.0, 3.0, 4.5])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, origin=origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -37.8056 == pytest.approx(log_p.item(), 0.0001)


def test_1rho2times():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0])
    delta = torch.tensor([1.5, 2.0])
    s = torch.zeros(2)

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)
    rho = torch.tensor([0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -78.4006528776 == pytest.approx(log_p.item(), 0.0001)


def test_1rho2times_grid():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0])
    delta = torch.tensor([1.5, 2.0])
    s = torch.zeros(2)

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)
    rho = torch.tensor([0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -78.4006528776 == pytest.approx(log_p.item(), 0.0001)


def test_1rho3times():
    sampling_times = torch.zeros(3)
    heights = torch.tensor([4.5, 5.5])

    R = torch.tensor([1.5, 4.0, 5.0])
    delta = torch.tensor([1.5, 2.0, 1.0])
    s = torch.zeros(3)

    lambda_, mu, psi = epidemiology_to_birth_death(R, delta, s)
    rho = torch.tensor([0.0, 0.0, 0.01])

    origin = torch.tensor([10.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -67.780094538296 == pytest.approx(log_p.item(), 0.0001)


def test_serial_1rho():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([2.0])
    mu = torch.tensor([1.0])
    psi = torch.tensor([0.5])

    origin = torch.tensor([6.0])

    bdsky = PiecewiseConstantBirthDeath(lambda_, mu, psi, origin=origin, survival=False)
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -19.0198 == pytest.approx(log_p.item(), 0.0001)


def test_serial_2rho():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([3.0, 2.0])
    mu = torch.tensor([2.5, 1.0])
    psi = torch.tensor([2.0, 0.5])

    origin = torch.tensor([6.0])
    # times = torch.cat((torch.tensor([0.0, 3.0]), origin))
    times = torch.tensor([0.0, 3.0])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, origin=origin, times=times, survival=False
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
    times = torch.tensor([0.0, 3.0, 4.5])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, rho=rho, origin=origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)))
    assert -37.8056 == pytest.approx(log_p.item(), 0.0001)


def test_serial_3rho_batch():
    sampling_times = torch.tensor([0.0, 1.0, 2.5, 3.5])
    heights = torch.tensor([2.0, 4.0, 5.0])

    lambda_ = torch.tensor([[3.0, 2.0, 4.0], [13.0, 12.0, 14.0]])
    mu = torch.tensor([[2.5, 1.0, 0.5], [12.5, 11.0, 10.5]])
    psi = torch.tensor([[2.0, 0.5, 1.0], [2.0, 0.5, 1.0]])

    origin = torch.tensor([6.0])
    times = torch.tensor([0.0, 3.0, 4.5])

    bdsky = PiecewiseConstantBirthDeath(
        lambda_, mu, psi, origin=origin, times=times, survival=False
    )
    log_p = bdsky.log_prob(torch.cat((sampling_times, heights)).repeat(2, 1))
    assert torch.allclose(log_p, torch.tensor([-37.8056, -75.5726318359375]))
