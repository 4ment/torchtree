import torch

from torchtree.distributions.transforms import CumSumExpTransform, LogTransform


def test_log_transform():
    t = LogTransform()
    x = torch.tensor([1.0, 2.0])
    y = t(x)
    assert torch.allclose(y, x.log())
    assert torch.allclose(x, t.inv(y))
    assert torch.allclose(t.log_abs_det_jacobian(x, y), -y)


def test_cum_sum_exp_transform():
    t = CumSumExpTransform()
    x = torch.tensor([0.1, -1.0, 10.0])
    y = t(x)
    assert torch.allclose(y, x.cumsum(0).exp())
    assert torch.allclose(x, t.inv(y))
