import numpy as np

from torchtree.evolution.datatype import (
    AminoAcidDataType,
    CodonDataType,
    GeneralDataType,
    NucleotideDataType,
)


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
    nuc = NucleotideDataType(None)
    gen = GeneralDataType('id', codes, ambiguities)

    assert gen.size == 1
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


def test_nucleotide():
    nuc_type = NucleotideDataType(None)
    assert nuc_type.size == 1
    assert nuc_type.state_count == 4
    assert len(nuc_type.states) == 4
    assert nuc_type.encoding('A') == 0
    assert nuc_type.encoding('T') == 3
    assert nuc_type.encoding('U') == 3
    assert nuc_type.partial('A') == (1.0,) + (0.0,) * 3
    assert nuc_type.partial('T') == (0.0,) * 3 + (1.0,)
    assert nuc_type.partial('-') == (1.0,) * 4


def test_amino_acid():
    aa_type = AminoAcidDataType(None)
    assert aa_type.size == 1
    assert aa_type.state_count == 20
    assert len(aa_type.states) == 20
    assert aa_type.encoding('A') == 0
    assert aa_type.encoding('Y') == 19
    assert aa_type.partial('A') == (1.0,) + (0.0,) * 19
    assert aa_type.partial('Y') == (0.0,) * 19 + (1.0,)
    assert aa_type.partial('-') == (1.0,) * 20
    assert aa_type.partial('/') == (1.0,) * 20


def test_codon():
    codon_type = CodonDataType(None, 'Universal')
    assert codon_type.size == 3
    assert codon_type.state_count == 61
    assert len(codon_type.states) == 61
    assert codon_type.encoding('AAA') == 0
    assert codon_type.encoding('TTT') == 60
    assert codon_type.partial('AAA') == (1.0,) + (0.0,) * 60
    assert codon_type.partial('TTT') == (0.0,) * 60 + (1.0,)
    assert codon_type.partial('---') == (1.0,) * 61
    assert codon_type.partial('///') == (1.0,) * 61
    # 49th triplet is a stop codon TAA
    assert codon_type.encoding('TAC') == 48
    # 51th triplet is a stop codon TAG
    assert codon_type.encoding('TAT') == 49
    # 57th triplet is a stop codon TGA
    assert codon_type.encoding('TGC') == 54
    assert codon_type.encoding('TGG') == 55
