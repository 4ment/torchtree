import pytest
import torch

from torchtree import Parameter
from torchtree.distributions.ctmc_scale import CTMCScale
from torchtree.evolution.tree_model import TimeTreeModel


@pytest.fixture
def tree_model_dict():
    tree_model = TimeTreeModel.json_factory(
        'tree',
        '((((A_0:1.5,B_1:0.5):2.5,C_2:2):2,D_3:3):10,E_12:4);',
        [1.5, 4.0, 6.0, 16.0],
        dict(zip(['A_0', 'B_1', 'C_2', 'D_3', 'E_12'], [0.0, 1.0, 2.0, 3.0, 12.0])),
    )
    return tree_model


def test_ctmc_scale(tree_model_dict):
    tree_model = TimeTreeModel.from_json(tree_model_dict, {})
    ctmc_scale = CTMCScale(None, Parameter(None, torch.tensor([0.001])), tree_model)
    assert 4.475351922659342 == pytest.approx(ctmc_scale().item(), 0.00001)


def test_ctmc_scale_batch(tree_model_dict):
    tree_model_dict['internal_heights']['tensor'] = [
        [1.5, 4.0, 6.0, 16.0],
        [1.5, 4.0, 6.0, 16.0],
    ]
    tree_model = TimeTreeModel.from_json(tree_model_dict, {})
    ctmc_scale = CTMCScale(
        None, Parameter(None, torch.tensor([[0.001], [0.001]])), tree_model
    )
    assert torch.allclose(torch.full((2, 1), 4.475351922659342), ctmc_scale())
