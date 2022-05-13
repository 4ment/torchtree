from __future__ import annotations

import abc

import numpy as np

from ..core.model import Identifiable
from ..core.utils import register_class
from ..typing import ID


class DataType(Identifiable, abc.ABC):
    @property
    @abc.abstractmethod
    def states(self) -> tuple[str, ...]:
        pass

    @property
    @abc.abstractmethod
    def state_count(self) -> int:
        pass

    @abc.abstractmethod
    def encoding(self, string: str) -> int:
        pass

    @abc.abstractmethod
    def partial(self, string: str, use_ambiguities=True) -> tuple[float, ...]:
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass


class AbstractDataType(DataType, abc.ABC):
    def __init__(self, id_: ID, states: tuple[str, ...]):
        super().__init__(id_)
        self._states = states
        self._state_count = len(states)
        self._size = len(states[0])

    @property
    def states(self) -> tuple[str, ...]:
        return self._states

    @property
    def state_count(self) -> int:
        return self._state_count

    @property
    def size(self) -> int:
        return self._size


@register_class
class NucleotideDataType(AbstractDataType):
    NUCLEOTIDES = "ACGTUKMRSWYBDHVN?-"
    # fmt: off
    NUCLEOTIDE_STATES = (17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                         # 16-31
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                         #                                           -  32-47
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                         #                                                ?  48-63
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16,
                         # @ A  B  C  D  e  f  G  H  i  j  K  l  M  N  o  64-79
                         17, 0, 11, 1, 12, 16, 16, 2, 13, 16, 16, 10, 16, 7, 15, 16,
                         # p  q  R  S  T  U  V  W  x  Y  z  80-95
                         16, 16, 5, 9, 3, 3, 14, 8, 16, 6, 16, 17, 17, 17, 17, 17,
                         # A  B  C  D  e  f  G  H  i  j  K  l  M  N  o   96-111
                         17, 0, 11, 1, 12, 16, 16, 2, 13, 16, 16, 10, 16, 7, 15, 16,
                         # p  q  R  S  T  U  V  W  x  Y  z  112-127
                         16, 16, 5, 9, 3, 3, 14, 8, 16, 6, 16, 17, 17, 17, 17, 17)
    # fmt: on

    NUCLEOTIDE_AMBIGUITY_STATES = (
        (1.0, 0.0, 0.0, 0.0),  # A
        (0.0, 1.0, 0.0, 0.0),  # C
        (0.0, 0.0, 1.0, 0.0),  # G
        (0.0, 0.0, 0.0, 1.0),  # T
        (0.0, 0.0, 0.0, 1.0),  # U
        (1.0, 0.0, 1.0, 0.0),  # R
        (0.0, 1.0, 0.0, 1.0),  # Y
        (1.0, 1.0, 0.0, 0.0),  # M
        (1.0, 0.0, 0.0, 1.0),  # W
        (0.0, 1.0, 1.0, 0.0),  # S
        (0.0, 0.0, 1.0, 1.0),  # K
        (0.0, 1.0, 1.0, 1.0),  # B
        (1.0, 0.0, 1.0, 1.0),  # D
        (1.0, 1.0, 0.0, 1.0),  # H
        (1.0, 1.0, 1.0, 0.0),  # V
        (1.0, 1.0, 1.0, 1.0),  # N
        (1.0, 1.0, 1.0, 1.0),  # ?
        (1.0, 1.0, 1.0, 1.0),  # -
    )

    def __init__(self, id_: ID):
        super().__init__(id_, ('A', 'C', 'G', 'T'))

    def encoding(self, string) -> int:
        return NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]

    def partial(self, string: str, use_ambiguities=True) -> tuple[float, ...]:
        if not use_ambiguities and string not in 'ACGTUacgtu':
            return (1.0,) * 4
        return NucleotideDataType.NUCLEOTIDE_AMBIGUITY_STATES[
            NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]
        ]

    @classmethod
    def from_json(cls, data, dic):
        return cls(data['id'])


@register_class
class AminoAcidDataType(AbstractDataType):
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBZX*?-"

    # fmt: off
    AMINO_ACIDS_STATES = (
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        #                                 *        -
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 23, 25, 25, 25, 25, 25,
        #                                                ?
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 24,
        #   A   B  C  D  E  F  G  H  I   j  K  L   M   N   o
        25, 0, 20, 1, 2, 3, 4, 5, 6, 7, 24, 8, 9, 10, 11, 24,
        # P  Q   R   S   T   u   V   W   X   Y   Z
        12, 13, 14, 15, 16, 24, 17, 18, 22, 19, 21, 25, 25, 25, 25, 25,
        # 	A   B  C  D  E  F  G  H  I   j  K  L   M   N   o
        25, 0, 20, 1, 2, 3, 4, 5, 6, 7, 24, 8, 9, 10, 11, 24,
        # P  Q   R   S   T   u   V   W   X   Y   Z
        12, 13, 14, 15, 16, 24, 17, 18, 22, 19, 21, 25, 25, 25, 25, 25)
    # fmt: on

    AMINO_ACIDS_AMBIGUITY_STATES = [[]] * 22
    for i in range(20):
        AMINO_ACIDS_AMBIGUITY_STATES[i] = [0.0] * 20
        AMINO_ACIDS_AMBIGUITY_STATES[i][i] = 1.0

    # B
    AMINO_ACIDS_AMBIGUITY_STATES[20] = [0.0] * 20
    AMINO_ACIDS_AMBIGUITY_STATES[20][AMINO_ACIDS_STATES[ord('D')]] = 1.0
    AMINO_ACIDS_AMBIGUITY_STATES[20][AMINO_ACIDS_STATES[ord('N')]] = 1.0
    # Z
    AMINO_ACIDS_AMBIGUITY_STATES[21] = [0.0] * 20
    AMINO_ACIDS_AMBIGUITY_STATES[21][AMINO_ACIDS_STATES[ord('E')]] = 1.0
    AMINO_ACIDS_AMBIGUITY_STATES[21][AMINO_ACIDS_STATES[ord('Q')]] = 1.0
    # X*?-
    AMINO_ACIDS_AMBIGUITY_STATES.extend([[1.0] * 20] * 4)

    AMINO_ACIDS_AMBIGUITY_STATES = [
        tuple(value) for i, value in enumerate(AMINO_ACIDS_AMBIGUITY_STATES)
    ]

    def __init__(self, id_: ID):
        super().__init__(id_, tuple(AminoAcidDataType.AMINO_ACIDS[:20]))

    def encoding(self, string) -> int:
        return AminoAcidDataType.AMINO_ACIDS_STATES[ord(string)]

    def partial(self, string: str, use_ambiguities=True) -> tuple[float, ...]:
        if (
            not use_ambiguities
            and string not in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy'
        ):
            return AminoAcidDataType.AMINO_ACIDS_AMBIGUITY_STATES[-1]
        return AminoAcidDataType.AMINO_ACIDS_AMBIGUITY_STATES[
            AminoAcidDataType.AMINO_ACIDS_STATES[ord(string)]
        ]

    @classmethod
    def from_json(cls, data, dic):
        return cls(data['id'])


@register_class
class CodonDataType(AbstractDataType):
    # Taken from BEAST GeneticCode.java

    GENETIC_CODE_TABLES = (
        # Universal
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        # Vertebrate Mitochondrial
        "KNKNTTTT*S*SMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Yeast
        "KNKNTTTTRSRSMIMIQHQHPPPPRRRRTTTTEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Mold Protozoan Mitochondrial
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Mycoplasma
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Invertebrate Mitochondrial
        "KNKNTTTTSSSSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Ciliate
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSS*CWCLFLF",
        # Echinoderm Mitochondrial
        "NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Euplotid Nuclear
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSCCWCLFLF",
        # Bacterial
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        # Alternative Yeast
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLSLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        # Ascidian Mitochondrial
        "KNKNTTTTGSGSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        # Flatworm Mitochondrial
        "NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYY*YSSSSWCWCLFLF",
        # Blepharisma Nuclear
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YQYSSSS*CWCLFLF",
        # No stops
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYYQYSSSSWCWCLFLF",
    )

    GENETIC_CODE_NAMES = (
        "Universal",
        "Vertebrate Mitochondrial",
        "Yeast",
        "Mold Protozoan Mitochondrial",
        "Mycoplasma",
        "Invertebrate Mitochondrial",
        "Ciliate",
        "Echinoderm Mitochondrial",
        "Euplotid Nuclear",
        "Bacterial",
        "Alternative Yeast",
        "Ascidian Mitochondrial",
        "Flatworm Mitochondrial",
        "Blepharisma Nuclear",
        "No stops",
    )

    NUMBER_OF_CODONS = (61, 60, 62, 62, 62, 62, 63, 62, 62, 61, 61, 62, 63, 62, 64)

    # fmt: off
    CODON_TRIPLETS = (
        "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT",
        "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT",
        "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
        "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT",
        "GAA", "GAC", "GAG", "GAT", "GCA", "GCC", "GCG", "GCT",
        "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
        "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT",
        "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT",
        "???", "---"
    )

    # fmt: on

    def __init__(self, id_: ID, genetic_code: str):
        index = [code.lower() for code in CodonDataType.GENETIC_CODE_NAMES].index(
            genetic_code.lower()
        )
        self.name = CodonDataType.GENETIC_CODE_NAMES[index]
        self._state_count = CodonDataType.NUMBER_OF_CODONS[index]
        self.table = CodonDataType.GENETIC_CODE_TABLES[index]
        self.triplets = CodonDataType.CODON_TRIPLETS
        nuc_type = NucleotideDataType(None)
        fn = (
            lambda codon: nuc_type.encoding(codon[0]) * 16
            + nuc_type.encoding(codon[1]) * 4
            + nuc_type.encoding(codon[2])
        )
        states = tuple(
            codon
            for i, codon in enumerate(self.triplets[:64])
            if self.table[fn(codon)] != '*'
        )
        self.stop_count = np.array([int(codon == '*') for codon in self.table]).cumsum()
        super().__init__(id_, states)

    def encoding(self, codon) -> int:
        n1 = NucleotideDataType.NUCLEOTIDE_STATES[ord(codon[0])]
        n2 = NucleotideDataType.NUCLEOTIDE_STATES[ord(codon[1])]
        n3 = NucleotideDataType.NUCLEOTIDE_STATES[ord(codon[2])]

        encoding = 65
        if n1 <= 3 and n2 <= 3 and n3 <= 3:
            encoding = n1 * 16 + n2 * 4 + n3
            encoding -= self.stop_count[encoding]
        return encoding

    def partial(self, string: str, use_ambiguities=True) -> tuple[float, ...]:
        encoding = self.encoding(string)
        if encoding == 65:
            p = [1.0] * self._state_count
        else:
            p = [0.0] * self._state_count
            p[encoding] = 1.0
        return tuple(p)

    @classmethod
    def from_json(cls, data, dic):
        code = data.get('genetic_code')
        return cls(data['id'], code)


@register_class
class GeneralDataType(AbstractDataType):
    def __init__(self, id_: ID, codes: tuple[str, ...], ambiguities: dict):
        super().__init__(id_, codes)
        self.codes = {code: idx for idx, code in enumerate(codes)}
        self._encoding = self.codes.copy()
        self.ambiguities = ambiguities
        for ambiguity in ambiguities.keys():
            self.codes[ambiguity] = np.array(
                [self.codes[s] for s in ambiguities[ambiguity]]
            )

            if (
                not isinstance(ambiguities[ambiguity], list)
                or len(ambiguities[ambiguity]) == 1
            ):
                # this is an alias for example {'U': 'T}
                self._encoding[ambiguity] = self.codes[ambiguities[ambiguity]]

    def encoding(self, string: str) -> int:
        return self._encoding.get(string, self.state_count)

    def partial(self, string: str, use_ambiguities=True) -> tuple[float, ...]:
        if string in self.codes:
            p = np.zeros(self.state_count)
            p[self.codes[string]] = 1.0
        else:
            p = np.ones(self.state_count)
        return tuple(p)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        codes = data['codes']
        ambiguities = data.get('ambiguities', {})
        return cls(id_, codes, ambiguities)
