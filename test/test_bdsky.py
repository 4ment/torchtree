import torch
import numpy as np
from phylotorch.evolution.bdsk import BDSKY
import pytest


def etest_BDSKY():
    # sampling_times = torch.tensor(np.array([0., 1., 2.5, 3.5]))
    # heights = torch.tensor(np.array([2., 4., 5.]))
    sampling_times = torch.tensor(np.array([0., 1., 6., 7]))
    heights = torch.tensor(np.array([6., 9., 10.]))
    times = torch.arange(0, 11) * 2.0
    death = np.arange(1, 11)[::-1]
    birth = np.full(10, 1.0)
    psi = 0.5
    rho = torch.full((10,), 0.0)

    R = torch.tensor(birth/(death+psi))
    delta = torch.tensor(death+psi)
    s = torch.tensor(psi/(death+psi))

    origin = torch.tensor([0.5])

    bdsky = BDSKY(R, delta, s, rho, sampling_times, origin, times=times)
    log_p = bdsky.log_prob(heights)
    print(log_p)
    exit(2)
    # a = ConstantCoalescent(torch.zeros((2, 4), dtype=torch.float64),
    #                                              torch.tensor(np.array([[2.], [30.]])))
    # samples = a.rsample()
    # print('samples', samples)
    # print('prob', a.log_prob(samples))
    # exit(2)
    # assert -13.295836866 == pytest.approx(log_p.item(), 0.0001)

def test_1rho():
    sampling_times = torch.tensor(np.array([0., 0., 0.]))
    heights = torch.tensor(np.array([4.5, 5.5]))

    R = torch.tensor(np.array([1.5]))  # effective reproductive number
    delta = torch.tensor(np.array([1.5]))  # total rate of becoming non infectious
    s = torch.zeros(1)  # probability of an individual being sampled
    rho = torch.tensor(np.array([0.01]))

    origin = torch.tensor([10.])

    bdsky = BDSKY(R, delta, s, rho, sampling_times, origin)
    log_p = bdsky.log_prob(heights)
    assert -8.520565 == pytest.approx(log_p.item(), 0.0001)
    print(log_p)
    # exit(2)

def test_1rho2times():
    sampling_times = torch.tensor(np.array([0., 0., 0.]))
    heights = torch.tensor(np.array([4.5, 5.5]))

    R = torch.tensor(np.array([1.5, 4.]))  # effective reproductive number
    delta = torch.tensor(np.array([1.5, 2.]))  # total rate of becoming non infectious
    s = torch.zeros(2)  # probability of an individual being sampled
    rho = torch.tensor(np.array([0., 0.01]))

    origin = torch.tensor([10.])

    bdsky = BDSKY(R, delta, s, rho, sampling_times, origin)
    log_p = bdsky.log_prob(heights)
    assert -78.4006528776 == pytest.approx(log_p.item(), 0.0001)
    print(log_p)