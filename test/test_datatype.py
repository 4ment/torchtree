import numpy as np

from torchtree.evolution.datatype import GeneralDataType, NucleotideDataType


def test_general():
    codes = 'A', 'C', 'G', 'T'
    ambiguities = {
        'U': 'T',
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'M': ['A', 'C'],
        'W': ['A', 'T'],
        'S': ['C', 'G'],
        'K': ['G', 'T'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G'],
    }
    nuc = NucleotideDataType()
    gen = GeneralDataType('id', codes, ambiguities)

    for code in codes:
        assert nuc.encoding(code) == gen.encoding(code)

    for code in ambiguities.keys():
        if nuc.encoding(code) < 4:
            assert nuc.encoding(code) == gen.encoding(code)
        else:
            assert gen.encoding(code) == 4

    for code in codes:
        np.testing.assert_array_equal(nuc.partial(code), gen.partial(code))

    for code in ambiguities.keys():
        np.testing.assert_array_equal(nuc.partial(code), gen.partial(code))
