from collections import OrderedDict

import pytest
import torch
import torch.distributions

from torchtree import Parameter
from torchtree.distributions.distributions import Distribution
from torchtree.distributions.joint_distribution import JointDistributionModel


def test_simple():
    normal = Distribution(
        None,
        torch.distributions.Normal,
        Parameter(None, torch.tensor([1.0])),
        OrderedDict(
            {
                'loc': Parameter(None, torch.tensor([0.0])),
                'scale': Parameter(None, torch.tensor([1.0])),
            }
        ),
    )

    exp = Distribution(
        None,
        torch.distributions.Exponential,
        Parameter(None, torch.tensor([1.0])),
        OrderedDict({'rate': Parameter(None, torch.tensor([1.0]))}),
    )
    joint = JointDistributionModel(None, [normal, exp])
    assert (-1.418939 - 1) == pytest.approx(joint().item())


def test_batch():
    normal = Distribution(
        None,
        torch.distributions.Normal,
        Parameter(None, torch.tensor([[1.0], [2.0]])),
        OrderedDict(
            {
                'loc': Parameter(None, torch.tensor([0.0])),
                'scale': Parameter(None, torch.tensor([1.0])),
            }
        ),
    )

    exp = Distribution(
        None,
        torch.distributions.Exponential,
        Parameter(None, torch.tensor([[1.0], [2.0]])),
        OrderedDict({'rate': Parameter(None, torch.tensor([1.0]))}),
    )
    joint = JointDistributionModel(None, [normal, exp])
    assert torch.allclose(torch.tensor([-1.418939 - 1, -2.918939 - 2]), joint())
