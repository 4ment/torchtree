import torch
from torch import Tensor


def soft_sort(s: Tensor, tau: float) -> Tensor:
    """Continuous relaxation for the argsort operator [#Prillo2020]_

    :param Tensor s: input tensor
    :param float tau: temperature
    :return: permutation matrix
    :rtype: Tensor

    .. [#Prillo2020] Prillo & Eisenschlos. SoftSort: A Continuous Relaxation for
    the argsort Operator. 2020.
    """
    s_sorted = s.sort(descending=True, dim=1)[0]
    pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
    return pairwise_distances.softmax(-1)


def soft_max(input: Tensor, k: float, dim, keepdim=False):
    return torch.logsumexp(input * k, dim=dim, keepdim=keepdim) / k
