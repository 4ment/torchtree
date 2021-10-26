import abc
from typing import List

import numpy as np

from ..core.model import Identifiable
from ..core.utils import register_class
from ..typing import ID


class DataType(abc.ABC):
    @property
    @abc.abstractmethod
    def state_count(self) -> int:
        pass

    @abc.abstractmethod
    def encoding(self, string: str) -> int:
        pass

    @abc.abstractmethod
    def partial(self, string: str, use_ambiguities=True) -> List[float]:
        pass


@register_class
class NucleotideDataType(DataType):
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

    @property
    def state_count(self) -> int:
        return 4

    def encoding(self, string) -> int:
        return NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]

    def partial(self, string: str, use_ambiguities=True) -> List[float]:
        if not use_ambiguities and string not in 'ACTGacgt':
            return [1.0] * 4
        return NucleotideDataType.NUCLEOTIDE_AMBIGUITY_STATES[
            NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]
        ]


@register_class
class GeneralDataType(Identifiable, DataType):
    def __init__(self, id_: ID, codes: dict, ambiguities: dict):
        super().__init__(id_)
        self.codes = {code: idx for idx, code in enumerate(codes)}
        self._encoding = self.codes.copy()
        self._state_count = len(codes)
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

    @property
    def state_count(self) -> int:
        return self._state_count

    def encoding(self, string: str) -> int:
        return self._encoding.get(string, self.state_count)

    def partial(self, string: str, use_ambiguities=True) -> List[float]:
        if string in self.codes:
            p = np.zeros(self.state_count)
            p[self.codes[string]] = 1.0
        else:
            p = np.ones(self.state_count)
        return list(p)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        codes = data['codes']
        ambiguities = data.get('ambiguities', {})
        return cls(id_, codes, ambiguities)
