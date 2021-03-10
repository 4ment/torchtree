import numpy as np
import pytest
import torch

from phylotorch.evolution.coalescent import ConstantCoalescent, PiecewiseConstantCoalescent, \
    PiecewiseConstantCoalescentGrid, ConstantCoalescentModel, PiecewiseConstantCoalescentModel, \
    PiecewiseConstantCoalescentGridModel


def inverse_transform_homochronous(ratios):
    heights = torch.zeros_like(ratios)
    heights[2] = ratios[-1]
    heights[1] = ratios[1] * heights[2].clone()
    heights[0] = ratios[0] * heights[1].clone()
    return heights


@pytest.fixture
def ratios_list():
    return 2. / 6., 6. / 12., 12.


@pytest.fixture
def node_heights_transformed(ratios_list):
    node_heights = {
        'id': 'node_heights',
        'type': 'phylotorch.core.model.TransformedParameter',
        'transform': 'phylotorch.evolution.tree_model.GeneralNodeHeightTransform',
        'parameters': {
            'tree': 'tree'
        },
        'x': [{
            'id': 'ratios',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': ratios_list[:-1]
        },
            {
                'id': 'root_height',
                'type': 'phylotorch.core.model.Parameter',
                'tensor': ratios_list[-1:]
            }
        ]
    }
    return node_heights


@pytest.fixture
def tree_model_node_heights_transformed(node_heights_transformed):
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': node_heights_transformed,
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "B", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "C", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
                {"id": "D", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}}
            ]
        }
    }
    return tree_model


def test_constant(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3.]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = ConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -13.295836866 == pytest.approx(log_p.item(), 0.0001)


def test_constant_json(tree_model_node_heights_transformed):
    example = {
        'id': 'coalescent',
        'type': 'phylotorch.evolution.coalescent.ConstantCoalescentModel',
        'theta': {
            'id': 'theta',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [3.]
        },
        'tree': tree_model_node_heights_transformed
    }
    constant = ConstantCoalescentModel.from_json(example, {})
    print(constant.tree.node_heights)
    assert -13.295836866 == pytest.approx(constant().item(), 0.0001)


def test_skyride(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3., 10., 4.]), requires_grad=True)
    heights = inverse_transform_homochronous(ratios)
    constant = PiecewiseConstantCoalescent(sampling_times, thetas)
    log_p = constant.log_prob(heights)
    assert -11.487491742782 == pytest.approx(log_p.item(), 0.0001)


def test_skyride_json(tree_model_node_heights_transformed):
    example = {
        'id': 'coalescent',
        'type': 'phylotorch.evolution.coalescent.PiecewiseConstantCoalescentModel',
        'theta': {
            'id': 'theta',
            'type': 'phylotorch.core.model.Parameter',
            'tensor': [3., 10., 4., 2., 3.]
        },
        'tree': tree_model_node_heights_transformed
    }
    skyride = PiecewiseConstantCoalescentModel.from_json(example, {})
    assert -11.487491742782 == pytest.approx(skyride().item(), 0.0001)


def test_skygrid(ratios_list):
    sampling_times = torch.tensor(np.array([0., 0., 0., 0.]))
    ratios = torch.tensor(np.array(ratios_list), requires_grad=True)
    thetas = torch.tensor(np.array([3., 10., 4., 2., 3.]), requires_grad=True)
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
            'tensor': [3., 10., 4., 2., 3.]
        },
        'tree': tree_model_node_heights_transformed,
        'cutoff': 10
    }
    skygrid = PiecewiseConstantCoalescentGridModel.from_json(example, {})
    assert -11.8751856 == pytest.approx(skygrid().item(), 0.0001)
