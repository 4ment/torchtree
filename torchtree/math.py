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
    s_sorted = s.sort(descending=True, dim=-1)[0]
    pairwise_distances = (s.unsqueeze(-2) - s_sorted.unsqueeze(-1)).abs().neg() / tau
    return pairwise_distances.softmax(-1)


def soft_max(input: Tensor, k: float, dim, keepdim=False):
    return torch.logsumexp(input * k, dim=dim, keepdim=keepdim) / k


def soft_searchsorted(sorted_sequence: Tensor, values: Tensor, tau):
    """Continuous relaxation for the torch.serachsorted function

    :param Tensor sorted_sequence: N-D or 1-D tensor, containing monotonically
        increasing sequence on the innermost dimension
    :param Tensor values: N-D tensor or a Scalar containing the search value(s)
    :param float tau: temperature
    :return: selection matrix
    :rtype: Tensor

    :example:
    >>> sorted_sequence = torch.tensor((-1., 10, 100.))
    >>> soft_searchsorted(sorted_sequence, torch.tensor(11.), 0.0001)
    tensor([[0., 0., 1., 0.]])
    >>> values = torch.tensor((0., 5000., 30.))
    >>> soft_selection = soft_searchsorted(sorted_sequence, values, 0.0001)
    >>> soft_selection
    tensor([[0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]])
    >>> indices = torch.searchsorted(sorted_sequence, values)
    >>> indices
    tensor([1, 3, 2])
    >>> torch.argmax(soft_selection, -1) == indices
    tensor([True, True, True])
    """
    sorted_sequence_bounded = torch.cat(
        (
            sorted_sequence[..., :1] - torch.finfo(sorted_sequence.dtype).eps,
            sorted_sequence,
        ),
        -1,
    )
    pairwise_distances = values.unsqueeze(-1) - sorted_sequence_bounded.unsqueeze(-2)
    return (pairwise_distances / tau).cumsum(-1).softmax(-1)
