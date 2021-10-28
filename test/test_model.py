import torch
import torch.distributions

from torchtree import CatParameter, Parameter, TransformedParameter, ViewParameter
from torchtree.core.parametric import ParameterListener


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


def test_parameter_to_dtype():
    p = Parameter('param', torch.tensor([1], dtype=torch.int32))
    assert p.dtype == torch.int32
    p.to(torch.float32)
    assert p.dtype == torch.float32


def test_view_parameter_repr():
    p = Parameter('param', torch.tensor([1, 2]))
    p2 = ViewParameter('p2', p, 1)
    assert eval(repr(p2)) == p2


def test_view_parameter():
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    p = Parameter('param', t)

    p1 = ViewParameter('param1', p, 3)
    assert torch.all(p1.tensor.eq(t[..., 3]))

    p1 = ViewParameter('param1', p, slice(3))
    assert torch.all(p1.tensor.eq(t[..., :3]))

    p1 = ViewParameter('param1', p, torch.tensor([0, 3]))
    assert torch.all(p1.tensor.eq(t[..., torch.tensor([0, 3])]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': 3,
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., 3]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': ':3',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., :3]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '2:',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., 2:]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '1:3',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., 1:3]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '1:4:2',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., 1:4:2]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '::-1',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(t[..., torch.LongTensor([3, 2, 1, 0])]))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '2::-1',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(torch.tensor(t.numpy()[Ellipsis, 2::-1].copy())))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': ':0:-1',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(torch.tensor(t.numpy()[..., :0:-1].copy())))

    p1 = ViewParameter.from_json(
        {
            'id': 'a',
            'type': 'torchtree.ViewParameter',
            'parameter': 'param',
            'indices': '2:0:-1',
        },
        {'param': p},
    )
    assert torch.all(p1.tensor.eq(torch.tensor(t.numpy()[..., 2:0:-1].copy())))


def test_view_parameter_listener():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = Parameter('param', t)

    p1 = ViewParameter('param1', p, 3)

    class FakeListener(ParameterListener):
        def __init__(self):
            self.gotit = False

        def handle_parameter_changed(self, variable, index, event):
            self.gotit = True

    listener = FakeListener()
    p1.add_parameter_listener(listener)

    p.fire_parameter_changed()
    assert listener.gotit is True

    listener.gotit = False
    p1.fire_parameter_changed()
    assert listener.gotit is True


def test_cat_parameter():
    t1 = torch.arange(4).view(2, 2)
    t2 = torch.arange(6).view(2, 3)
    param = CatParameter('param', (Parameter(None, t1), Parameter(None, t2)), -1)
    assert torch.all(torch.cat((t1, t2), -1) == param.tensor)
    assert eval(repr(param)) == param
