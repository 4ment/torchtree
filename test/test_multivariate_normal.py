import pytest
import torch
import torch.distributions.multivariate_normal

from torchtree import Parameter
from torchtree.distributions.multivariate_normal import MultivariateNormal


@pytest.mark.parametrize(
    "parameterization,param",
    [
        ("scale_tril", [[1.0, 0.0], [2.0, 3.0]]),
        ('precision_matrix', [[2.0, 0.0], [0.0, 3.0]]),
        ('covariance_matrix', [[2.0, 0.0], [0.0, 3.0]]),
    ],
)
def test_parameterization(parameterization, param):
    loc = torch.tensor([1.0, 2.0])
    x = torch.tensor([1.0, 2.0])
    param_tensor = torch.tensor(param)
    kwargs = {parameterization: Parameter(None, param_tensor)}
    dist_model = MultivariateNormal(
        None, Parameter(None, x), Parameter(None, loc), **kwargs
    )

    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        loc, **{parameterization: param_tensor}
    )
    assert dist.log_prob(x) == dist_model()

    x_tuple = (Parameter(None, x[:-1]), Parameter(None, x[-1:]))
    dist_model = MultivariateNormal(None, x_tuple, Parameter(None, loc), **kwargs)
    assert dist.log_prob(x) == dist_model()


@pytest.mark.parametrize(
    "parameterization,param",
    [
        ("scale_tril", [[1.0, 0.0], [2.0, 3.0]]),
        ('precision_matrix', [[2.0, 0.0], [0.0, 3.0]]),
        ('covariance_matrix', [[2.0, 0.0], [0.0, 3.0]]),
    ],
)
def test_parameterization_rsample(parameterization, param):
    torch.manual_seed(0)
    loc = torch.tensor([1.0, 2.0])
    x = torch.tensor([1.0, 2.0])
    param_tensor = torch.tensor(param)
    kwargs = {parameterization: Parameter(None, param_tensor)}
    dist_model = MultivariateNormal(
        None, Parameter(None, x), Parameter(None, loc), **kwargs
    )
    dist_model.rsample()

    torch.manual_seed(0)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        loc, **{parameterization: param_tensor}
    )
    samples = dist.rsample()
    assert torch.all(samples == dist_model.x.tensor)

    torch.manual_seed(0)
    x_tuple = (Parameter(':-1', x[:-1]), Parameter('-1:', x[-1:]))
    dist_model = MultivariateNormal(None, x_tuple, Parameter(None, loc), **kwargs)
    dist_model.rsample()
    assert torch.all(samples == dist_model.x.tensor)
