import numpy as np
import pytest
import torch

from phylotorch.coalescent import ConstantCoalescent, PiecewiseConstantCoalescent, PiecewiseConstantCoalescentGrid


def inverse_transform_homochronous(ratios):
    heights = torch.zeros_like(ratios)
    heights[2] = ratios[-1]
    heights[1] = ratios[1] * heights[2].clone()
    heights[0] = ratios[0] * heights[1].clone()
    return heights


@pytest.fixture
def ratios_list():
    return 2. / 6., 6. / 12., 12.


def test_constant(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3.]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = ConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -13.295836866 == pytest.approx(log_p.item(), 0.0001)


def test_skyride(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3., 10., 4.]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = PiecewiseConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -11.487491742782 == pytest.approx(log_p.item(), 0.0001)


def test_skygrid(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3., 10., 4., 2., 3.]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    grid = torch.tensor(np.linspace(0, 10.0, num=5)[1:])
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert -11.8751856 == pytest.approx(log_p.item(), 0.0001)
