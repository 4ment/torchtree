import abc

import numpy as np

from phylotorch.core.model import Identifiable


class DataType(abc.ABC):

    @property
    @abc.abstractmethod
    def state_count(self):
        pass

    @abc.abstractmethod
    def encoding(self, string):
        pass

    @abc.abstractmethod
    def partial(self, string):
        pass


class NucleotideDataType(DataType):
    NUCLEOTIDES = "ACGTUKMRSWYBDHVN?-"
    NUCLEOTIDE_STATES = (17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,  # 0-15
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,  # 16-31
                         #                                           -
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,  # 32-47
                         #                                                ?
                         17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16,  # 48-63
                         #	    A  B  C  D  e  f  G  H  i  j  K  l  M  N  o
                         17, 0, 11, 1, 12, 16, 16, 2, 13, 16, 16, 10, 16, 7, 15, 16,  # 64-79
                         #	 p  q  R  S  T  U  V  W  x  Y  z
                         16, 16, 5, 9, 3, 3, 14, 8, 16, 6, 16, 17, 17, 17, 17, 17,  # 80-95
                         #	    A  B  C  D  e  f  G  H  i  j  K  l  M  N  o
                         17, 0, 11, 1, 12, 16, 16, 2, 13, 16, 16, 10, 16, 7, 15, 16,  # 96-111
                         #	 p  q  R  S  T  U  V  W  x  Y  z
                         16, 16, 5, 9, 3, 3, 14, 8, 16, 6, 16, 17, 17, 17, 17, 17)  # 112-127

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
    def state_count(self):
        return 4

    def encoding(self, string):
        return NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]

    def partial(self, string):
        return np.array(
            NucleotideDataType.NUCLEOTIDE_AMBIGUITY_STATES[NucleotideDataType.NUCLEOTIDE_STATES[ord(string)]])


class GeneralDataType(Identifiable, DataType):

    def __init__(self, id_, codes, ambiguities):
        self.codes = {code: idx for idx, code in enumerate(codes)}
        self._encoding = self.codes.copy()
        self._state_count = len(codes)
        self.ambiguities = ambiguities
        for ambiguity in ambiguities.keys():
            self.codes[ambiguity] = np.array([self.codes[s] for s in ambiguities[ambiguity]])

            if not isinstance(ambiguities[ambiguity], list) or len(ambiguities[ambiguity]) == 1:
                # this is an alias for example {'U': 'T}
                self._encoding[ambiguity] = self.codes[ambiguities[ambiguity]]
        super(GeneralDataType, self).__init__(id_)

    @property
    def state_count(self):
        return self._state_count

    def encoding(self, string):
        return self._encoding.get(string, self.state_count)

    def partial(self, string):
        if string in self.codes:
            p = np.zeros(self.state_count)
            p[self.codes[string]] = 1.0
        else:
            p = np.ones(self.state_count)
        return p

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        codes = data['codes']
        ambiguities = data.get('ambiguities', {})
        return cls(id_, codes, ambiguities)
