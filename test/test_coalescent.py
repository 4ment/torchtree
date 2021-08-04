import numpy as np
import pytest
import torch

from phylotorch import Parameter
from phylotorch.evolution.coalescent import (
    ConstantCoalescent,
    ConstantCoalescentModel,
    PiecewiseConstantCoalescent,
    PiecewiseConstantCoalescentGrid,
    PiecewiseConstantCoalescentGridModel,
    PiecewiseConstantCoalescentModel,
)
from phylotorch.evolution.tree_model import TimeTreeModel


def inverse_transform_homochronous(ratios):
    heights = torch.zeros_like(ratios)
    heights[..., 2] = ratios[..., -1]
    heights[..., 1] = ratios[..., 1] * heights[..., 2].clone()
    heights[..., 0] = ratios[..., 0] * heights[..., 1].clone()
    return heights


@pytest.fixture
def ratios_list():
    return 2.0 / 6.0, 6.0 / 12.0, 12.0


@pytest.fixture
def node_heights_transformed(ratios_list):
    node_heights = {
        'id': 'node_heights',
        'type': 'phylotorch.core.model.TransformedParameter',
        'transform': 'phylotorch.evolution.tree_model.GeneralNodeHeightTransform',
        'parameters': {'tree': 'tree'},
        'x': [
            {
                'id': 'ratios',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': ratios_list[:-1],
            },
            {
                'id': 'root_height',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': ratios_list[-1:],
            },
        ],
    }
    return node_heights


@pytest.fixture
def tree_model_node_heights_transformed(node_heights_transformed):
    tree_model = TimeTreeModel.json_factory(
        'tree',
        '(((A,B),C),D);',
        dict(zip('ABCD', [0.0, 0.0, 0.0, 0.0])),
        **{'node_heights': node_heights_transformed}
    )
    return tree_model


def test_constant(ratios_list):
    sampling_times = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3.0]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = ConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert torch.allclose(torch.tensor([-13.295836866], dtype=log_p.dtype), log_p)


def test_constant_batch(ratios_list):
    ratios_list = list(ratios_list) + [2.0 * v for v in ratios_list]
    sampling_times = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]))
    ratios = torch.tensor(np.array(ratios_list).reshape(2, 3), requires_grad=True)
    thetas = torch.tensor(np.array([[3.0], [6.0]]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = ConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert torch.allclose(
        torch.tensor([[-13.295836866], [-25.375278407684164]], dtype=log_p.dtype), log_p
    )


def test_constant_json(tree_model_node_heights_transformed):
    example = {
        'id': 'coalescent',
        'type': 'phylotorch.evolution.coalescent.ConstantCoalescentModel',
        'theta': {
            'id': 'theta',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [3.0],
        },
        'tree_model': tree_model_node_heights_transformed,
    }
    constant = ConstantCoalescentModel.from_json(example, {})
    assert -13.295836866 == pytest.approx(constant().item(), 0.0001)


def test_skyride(ratios_list):
    sampling_times = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3.0, 10.0, 4.0]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = PiecewiseConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -11.487491742782 == pytest.approx(log_p.item(), 0.0001)


def test_skyride_batch(ratios_list):
    sampling_times = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]))
    ratios = torch.tensor(np.array([ratios_list] + [ratios_list]), requires_grad=True)
    thetas = torch.tensor(
        np.array([[3.0, 10.0, 4.0], [3.0, 10.0, 4.0]]), requires_grad=True
    )
    heights = inverse_transform_homochronous(ratios)
    constant = PiecewiseConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert torch.allclose(
        torch.tensor([[-11.487491742782], [-11.487491742782]], dtype=thetas.dtype),
        log_p,
    )


def test_skyride_heterochronous():
    sampling_times = torch.tensor(np.array([0.0, 1.0, 1.0, 0.0]))
    thetas = torch.tensor(np.array([3.0, 10.0, 4.0]), requires_grad=True)
    heights = torch.tensor(np.array([3.0, 2.0, 4.0]))
    constant = PiecewiseConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -7.67082507611538 == pytest.approx(log_p.item())


def test_skyride_json(tree_model_node_heights_transformed):
    example = {
        'id': 'coalescent',
        'type': 'phylotorch.evolution.coalescent.PiecewiseConstantCoalescentModel',
        'theta': {
            'id': 'theta',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [3.0, 10.0, 4.0],
        },
        'tree_model': tree_model_node_heights_transformed,
    }
    skyride = PiecewiseConstantCoalescentModel.from_json(example, {})
    assert -11.487491742782 == pytest.approx(skyride().item(), 0.0001)


def test_skygrid(ratios_list):
    sampling_times = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3.0, 10.0, 4.0, 2.0, 3.0]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    grid = torch.tensor(np.linspace(0, 10.0, num=5)[1:])
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert -11.8751856 == pytest.approx(log_p.item(), 0.0001)


def test_skygrid_json(tree_model_node_heights_transformed):
    example = {
        'id': 'coalescent',
        'type': 'phylotorch.evolution.coalescent.PiecewiseConstantCoalescentGridModel',
        'theta': {
            'id': 'theta',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [3.0, 10.0, 4.0, 2.0, 3.0],
        },
        'tree_model': tree_model_node_heights_transformed,
        'cutoff': 10,
    }
    skygrid = PiecewiseConstantCoalescentGridModel.from_json(example, {})
    assert -11.8751856 == pytest.approx(skygrid().item(), 0.0001)


@pytest.mark.parametrize(
    "cutoff,expected", [(10.0, -19.594893640219844), (18.0, -14.918634593243764)]
)
def test_skygrid_heterochronous(cutoff, expected):
    sampling_times = torch.tensor(np.array([0.0, 1.0, 2.0, 3.0, 12.0]))
    thetas_log = torch.tensor(np.array([1.0, 3.0, 6.0, 8.0, 9.0]))
    thetas = thetas_log.exp()
    heights = torch.tensor(np.array([1.5, 4.0, 6.0, 16.0]))
    grid = torch.linspace(0, cutoff, steps=5)[1:]
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert torch.allclose(torch.tensor([expected], dtype=log_p.dtype), log_p)


def test_skygrid_heterochronous_batch():
    sampling_times = torch.tensor(np.array([0.0, 1.0, 2.0, 3.0, 12.0]))
    thetas_log = torch.tensor(
        np.array([[1.0, 3.0, 6.0, 8.0, 9.0], [1.0, 3.0, 6.0, 8.0, 9.0]])
    )
    thetas = thetas_log.exp()
    heights = torch.tensor(np.array([[1.5, 4.0, 6.0, 16.0], [1.5, 4.0, 6.0, 26.0]]))
    grid = torch.linspace(0, 10.0, steps=5)[1:]
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert torch.allclose(
        torch.tensor(
            [[-19.594893640219844], [-19.596127738260712]], dtype=thetas.dtype
        ),
        log_p,
    )


def test_skygrid_heterochronous_batch_times():
    sampling_times = torch.tensor(np.array([0.0, 1.0, 2.0, 3.0, 12.0]))
    thetas_log = torch.tensor(np.array([1.0, 3.0, 6.0, 8.0, 9.0]))
    thetas = thetas_log.exp()
    heights = torch.tensor(np.array([[1.5, 4.0, 6.0, 16.0], [1.5, 4.0, 6.0, 26.0]]))
    grid = torch.linspace(0, 10.0, steps=5)[1:]
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert torch.allclose(
        torch.tensor(
            [[-19.594893640219844], [-19.596127738260712]], dtype=thetas.dtype
        ),
        log_p,
    )


def test_skygrid_heterochronous_batch_thetas():
    sampling_times = torch.tensor(np.array([0.0, 1.0, 2.0, 3.0, 12.0]))
    thetas_log = torch.tensor(
        np.array([[1.0, 3.0, 6.0, 8.0, 9.0], [1.0, 3.0, 6.0, 8.0, 9.0]])
    )
    thetas = thetas_log.exp()
    heights = torch.tensor(np.array([1.5, 4.0, 6.0, 16.0]))
    grid = torch.linspace(0, 10.0, steps=5)[1:]
    constant = PiecewiseConstantCoalescentGrid(sampling_times, thetas, grid)
    log_p = constant.log_prob(heights)
    assert torch.allclose(
        torch.tensor(
            [[-19.594893640219844], [-19.594893640219844]], dtype=thetas.dtype
        ),
        log_p,
    )


def test_skygrid_heterochronous_2_trees():
    sampling_times = torch.tensor(np.array([0.0, 1.0, 2.0, 3.0, 12.0]))
    thetas_log = torch.tensor(np.array([1.0, 3.0, 6.0, 8.0, 9.0]))
    thetas = Parameter(None, thetas_log.exp())
    grid = Parameter(None, torch.linspace(0, 10.0, steps=5)[1:])
    constant = PiecewiseConstantCoalescentGridModel(
        None,
        thetas,
        [
            Parameter(None, torch.tensor([1.5, 4.0, 6.0, 16.0])),
            Parameter(None, torch.tensor([1.5, 4.0, 6.0, 26.0])),
        ],
        sampling_times,
        grid,
    )
    assert torch.allclose(
        torch.tensor(
            [[-19.594893640219844], [-19.596127738260712]], dtype=thetas.dtype
        ).sum(0),
        constant(),
    )
