import numpy as np
import pytest
import torch

from torchtree import Parameter
from torchtree.evolution.datatype import CodonDataType
from torchtree.evolution.substitution_model import (
    GTR,
    HKY,
    JC69,
    LG,
    MG94,
    GeneralSymmetricSubstitutionModel,
)

# r=c(0.060602,0.402732,0.028230,0.047910,0.407249,0.053277)
# f=c(0.479367,0.172572,0.140933,0.207128)
# R=matrix(c(0,r[1],r[2],r[3],
#        r[1],0,r[4],r[5],
#        r[2],r[4],0,r[6],
#        r[3],r[5],r[6],0),nrow=4)
# Q=R %*% diag(f,4,4)
# diag(Q)=-apply(Q, 1, sum)
# Q=-Q/sum(diag(Q)*f)
# e=eigen(Q)
# e$vectors %*% diag(exp(e$values*0.1)) %*% solve(e$vectors)


def test_general_symmetric():
    rates = torch.tensor(
        np.array([0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277])
    )
    pi = torch.tensor(np.array([0.479367, 0.172572, 0.140933, 0.207128]))
    mapping = torch.arange(6)
    subst_model = GeneralSymmetricSubstitutionModel(
        'gen',
        Parameter('mapping', mapping),
        Parameter('rates', rates),
        Parameter('pi', pi),
    )
    P = subst_model.p_t(torch.tensor(np.array([0.1])))

    gtr = GTR('gtr', Parameter('rates', rates), Parameter('pi', pi))
    P_gtr = gtr.p_t(torch.tensor(np.array([0.1])))

    np.testing.assert_allclose(P, P_gtr, rtol=1e-06)


def test_GTR():
    rates = torch.tensor(
        np.array([0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277])
    )
    pi = torch.tensor(np.array([0.479367, 0.172572, 0.140933, 0.207128]))
    subst_model = GTR('gtr', Parameter('rates', rates), Parameter('pi', pi))
    P = subst_model.p_t(torch.tensor(np.array([[0.1]])))
    P_expected = torch.tensor(
        np.array(
            [
                [0.93717830, 0.009506685, 0.047505899, 0.005809115],
                [0.02640748, 0.894078744, 0.006448058, 0.073065722],
                [0.16158572, 0.007895626, 0.820605951, 0.009912704],
                [0.01344433, 0.060875872, 0.006744752, 0.918935042],
            ]
        )
    )
    assert torch.allclose(P.squeeze(), P_expected, rtol=1e-05)

    subst_model = GTR.from_json(
        {
            'id': 'gtr',
            'type': 'torchtree.evolution.substitution_model.GTR',
            'rates': {
                'id': 'rates',
                'type': 'torchtree.Parameter',
                'tensor': [0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277],
                'dtype': 'torch.float64',
            },
            'frequencies': {
                'id': 'pi',
                'type': 'torchtree.Parameter',
                'tensor': [0.479367, 0.172572, 0.140933, 0.207128],
                'dtype': 'torch.float64',
            },
        },
        {},
    )
    P = subst_model.p_t(torch.tensor([[0.1]]))
    assert torch.allclose(P.squeeze(), P_expected, rtol=1e-05)


def test_GTR_batch():
    rates = torch.tensor(
        np.array(
            [
                [0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277],
                [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
            ]
        )
    )
    pi = torch.tensor(
        np.array(
            [
                [0.479367, 0.172572, 0.140933, 0.207128],
                [0.479367, 0.172572, 0.140933, 0.207128],
            ]
        )
    )
    subst_model = GTR('gtr', Parameter('rates', rates), Parameter('pi', pi))
    P = subst_model.p_t(torch.tensor(np.array([[0.1], [0.001]])))
    P_expected = torch.tensor(
        np.array(
            [
                [
                    [
                        [0.93717830, 0.009506685, 0.047505899, 0.005809115],
                        [0.02640748, 0.894078744, 0.006448058, 0.073065722],
                        [0.16158572, 0.007895626, 0.820605951, 0.009912704],
                        [0.01344433, 0.060875872, 0.006744752, 0.918935042],
                    ]
                ],
                [
                    [
                        [0.9992649548, 0.0001581235, 0.0003871353, 0.0001897863],
                        [0.0004392323, 0.9988625812, 0.0001291335, 0.0005690531],
                        [0.0013167952, 0.0001581235, 0.9983352949, 0.0001897863],
                        [0.0004392323, 0.0004741156, 0.0001291335, 0.9989575186],
                    ]
                ],
            ]
        )
    )
    assert torch.allclose(P, P_expected, rtol=1e-05)


@pytest.fixture
def hky_fixture():
    kappa = torch.tensor([3.0])
    pi = torch.tensor([0.479367, 0.172572, 0.140933, 0.207128])
    branch_lengths = torch.tensor([[0.1], [0.001]])
    P_expected = torch.tensor(
        [
            [
                [0.93211187, 0.01511617, 0.03462891, 0.01814305],
                [0.04198939, 0.89405292, 0.01234480, 0.05161289],
                [0.11778615, 0.01511617, 0.84895463, 0.01814305],
                [0.04198939, 0.04300210, 0.01234480, 0.90266370],
            ],
            [
                [0.9992649548, 0.0001581235, 0.0003871353, 0.0001897863],
                [0.0004392323, 0.9988625812, 0.0001291335, 0.0005690531],
                [0.0013167952, 0.0001581235, 0.9983352949, 0.0001897863],
                [0.0004392323, 0.0004741156, 0.0001291335, 0.9989575186],
            ],
        ]
    ).unsqueeze(1)
    return kappa, pi, P_expected, branch_lengths


def test_HKY(hky_fixture):
    # r=c(1,3,1,1,3,1)
    kappa, pi, hky_P_expected, branch_lengths = hky_fixture
    subst_model = HKY('hky', Parameter('kappa', kappa), Parameter('pi', pi))
    P = subst_model.p_t(branch_lengths)
    assert torch.allclose(P, hky_P_expected, atol=1e-06)

    P = super(type(subst_model), subst_model).p_t(branch_lengths)
    assert torch.allclose(P, hky_P_expected, atol=1e-06)


def test_HKY_json(hky_fixture):
    kappa, pi, hky_P_expected, branch_lengths = hky_fixture

    subst_model = HKY.from_json(
        {
            'id': 'hky',
            'type': 'torchtree.evolution.substitution_model.HKY',
            'kappa': {
                'id': 'kappa',
                'type': 'torchtree.Parameter',
                'tensor': kappa.tolist(),
            },
            'frequencies': {
                'id': 'pi',
                'type': 'torchtree.Parameter',
                'tensor': pi.tolist(),
            },
        },
        {},
    )
    P = subst_model.p_t(branch_lengths)
    assert torch.allclose(P, hky_P_expected, atol=1e-06)


def test_HKY_batch(hky_fixture):
    kappa, pi, hky_P_expected, branch_lengths = hky_fixture
    kappa = Parameter(
        None,
        torch.cat((torch.tensor([2.0], dtype=kappa.dtype), kappa), 0).unsqueeze(-1),
    )
    frequencies = Parameter(
        None, torch.cat((torch.full_like(pi, 0.25), pi), 0).reshape((2, 4))
    )
    hky = HKY('hky', kappa, frequencies)
    probs = hky.p_t(
        torch.tensor([[5, 10.0, 20.0], [0.5, 0.1, 0.001]], dtype=kappa.dtype).unsqueeze(
            -1
        )
    )
    assert torch.allclose(probs[1, 1:3], hky_P_expected, atol=1e-06)


@pytest.mark.parametrize("t", [0.001, 0.1])
def test_JC69(t):
    ii = 1.0 / 4.0 + 3.0 / 4.0 * np.exp(-4.0 / 3.0 * t)
    ij = 1.0 / 4.0 - 1.0 / 4.0 * np.exp(-4.0 / 3.0 * t)
    subst_model = JC69('jc')
    P = torch.squeeze(subst_model.p_t(torch.tensor(np.array([t]))))
    assert ii == pytest.approx(P[0, 0].item(), 1.0e-6)
    assert ij == pytest.approx(P[0, 1].item(), 1.0e-6)

    subst_model = JC69.from_json(
        {'id': 'jc', 'type': 'torchtree.substitution_model.JC69'}, {}
    )
    P = torch.squeeze(subst_model.p_t(torch.tensor(np.array([t]))))
    assert ii == pytest.approx(P[0, 0].item(), 1.0e-6)
    assert ij == pytest.approx(P[0, 1].item(), 1.0e-6)


def test_JC69_batch():
    jc = JC69('jc')
    probs = jc.p_t(torch.tensor([[[0.5, 1.0, 2.0]], [[5, 10.0, 20.0]]]))
    assert probs.shape == torch.Size([2, 1, 3, 4, 4])


def test_general_GTR():
    rates = torch.tensor(
        np.array([0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277])
    )
    pi = torch.tensor(np.array([0.479367, 0.172572, 0.140933, 0.207128]))
    mapping = torch.arange(6, dtype=torch.long)
    subst_model = GeneralSymmetricSubstitutionModel(
        'gen',
        Parameter('mapping', mapping),
        Parameter('rates', rates),
        Parameter('pi', pi),
    )
    P = subst_model.p_t(torch.tensor(np.array([[0.1]])))

    subst_model2 = GTR('gtr', Parameter('rates', rates), Parameter('pi', pi))
    P_expected = subst_model2.p_t(torch.tensor(np.array([[0.1]])))
    np.testing.assert_allclose(P, P_expected, rtol=1e-06)


@pytest.mark.parametrize(
    "input,size",
    [
        ([[1.0]], [1, 1, 20, 20]),
        ([[1.0, 1.0]], [1, 2, 20, 20]),
        ([[1.0], [1.0]], [2, 1, 20, 20]),
        ([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [3, 2, 20, 20]),
    ],
)
def test_amino_acid(input, size):
    subst_model = LG(None)
    matrices = subst_model.p_t(torch.tensor(input))
    assert matrices.shape == torch.Size(size)


def test_MG94():
    codon_type = CodonDataType(None, 'Universal')
    subst_model = MG94(
        'mg94',
        codon_type,
        Parameter('alpha', torch.tensor([[11.0], [1.0]])),
        Parameter('beta', torch.tensor([[12.0], [1.0]])),
        Parameter('kappa', torch.tensor([[13.0], [1.0]])),
        Parameter('frequencies', torch.full((61,), 1.0 / 61)),
    )
    Q = subst_model.q()
    assert torch.allclose(Q[1, 0, 1:], torch.full((60,), 1 / 61))
    assert torch.allclose(Q[1, range(61), range(61)], torch.full((61,), -60 / 61))
