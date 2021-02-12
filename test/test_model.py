import torch
import torch.distributions

from phylotorch.core.model import Parameter, TransformedParameter


def test_parameter_repr():
    p = Parameter('param', torch.tensor([1, 2]))
    assert eval(repr(p)) == p


def test_transformed_parameter():
    t = torch.tensor([1.0, 2.0])
    p1 = Parameter('param', t)
    transformed = torch.distributions.ExpTransform()
    p2 = TransformedParameter('transformed', p1, transformed)
    assert p2.need_update is False
    assert torch.all(p2.tensor.eq(t.exp()))
    assert p2.need_update is False

    p1.tensor = torch.tensor([1.0, 3.0])
    assert p2.need_update is True

    assert torch.all(p2.tensor.eq(p1.tensor.exp()))
    assert p2.need_update is False

    # jacobian
    p1.tensor = t
    assert p2.need_update is True
    assert torch.all(p2().eq(p1.tensor))
    assert p2.need_update is False

    assert torch.all(p2.tensor.eq(p1.tensor.exp()))
