from __future__ import annotations

import collections
import itertools

import numpy as np

from ..core.model import Identifiable
from ..core.utils import process_object, register_class
from ..typing import ID
from .datatype import DataType, NucleotideDataType
from .taxa import Taxa

Sequence = collections.namedtuple('Sequence', ['taxon', 'sequence'])


@register_class
class Alignment(Identifiable, collections.UserList):
    """Sequence alignment.

    :param id_: ID of object
    :param sequences: list of sequences
    :param taxa: Taxa object
    """

    _tag = 'alignment'

    def __init__(
        self, id_: ID, sequences: list[Sequence], taxa: Taxa, data_type: DataType
    ) -> None:
        self._sequence_size = len(sequences[0].sequence)
        self._taxa = taxa
        self._data_type = data_type
        indexing = {taxon.id: idx for idx, taxon in enumerate(taxa)}
        sequences.sort(key=lambda x: indexing[x.taxon])
        Identifiable.__init__(self, id_)
        collections.UserList.__init__(self, sequences)

    @property
    def sequence_size(self) -> int:
        return self._sequence_size

    @property
    def taxa(self) -> Taxa:
        return self._taxa

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @classmethod
    def get(cls, id_: ID, filename: str, taxa: Taxa) -> Alignment:
        sequences = read_fasta_sequences(filename)
        return cls(id_, sequences, taxa)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)

        if isinstance(data['datatype'], str) and data['datatype'] == 'nucleotide':
            if 'nucleotide' in dic and isinstance(
                dic['nucleotide'], NucleotideDataType
            ):
                data_type = process_object(data['datatype'], dic)
            else:
                data_type = NucleotideDataType(None)
        else:
            data_type = process_object(data['datatype'], dic)

        if 'sequences' in data:
            sequences = []
            for sequence in data['sequences']:
                sequences.append(Sequence(sequence['taxon'], sequence['sequence']))
        elif 'file' in data:
            sequences = read_fasta_sequences(data['file'])
        else:
            raise ValueError('sequences or file should be specified in Alignment')
        return cls(id_, sequences, taxa, data_type)


def read_fasta_sequences(filename: str) -> list[Sequence]:
    sequences = {}
    with open(filename, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('>'):
                taxon = line[1:]
                sequences[taxon] = ''
            else:
                sequences[taxon] += line
    return [Sequence(taxon, sequence) for taxon, sequence in sequences.items()]


def calculate_frequencies(alignment: Alignment):
    data_type = alignment.data_type
    frequencies = collections.Counter()
    for sequence in alignment:
        freqs = collections.Counter(
            map(
                data_type.encoding,
                map(''.join, zip(*[iter(sequence.sequence)] * data_type.size)),
            )
        )
        frequencies.update(freqs)
    frequencies = np.array(list(map(lambda x: x[1], sorted(frequencies.items()))))[
        : data_type.state_count
    ]
    return (frequencies / frequencies.sum()).tolist()


def calculate_frequencies_per_codon_position(alignment: Alignment):
    data_type = NucleotideDataType('temp')
    frequencies = np.zeros(18 * 3)
    for sequence in alignment:
        for i in range(3):
            freqs = collections.Counter(sequence.sequence[i::3])
            for nuc, count in freqs.items():
                encoding = data_type.encoding(nuc)
                frequencies[18 * i + encoding] += count
    return [
        (
            frequencies[(i * 18) : (i * 18 + 4)]
            / frequencies[(i * 18) : (i * 18 + 4)].sum()
        ).tolist()
        for i in range(3)
    ]


def calculate_F3x4_from_nucleotide(data_type, nuc_freq):
    codon_freqs = np.full(64, 0.0)
    coding_indices = [i for i, value in enumerate(data_type.table[:64]) if value != '*']
    for n1, n2, n3 in itertools.product((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)):
        codon_freqs[n1 * 16 + n2 * 4 + n3] = (
            nuc_freq[0][n1] * nuc_freq[1][n2] * nuc_freq[2][n3]
        )
    codon_freqs = codon_freqs[coding_indices]
    return (codon_freqs / codon_freqs.sum()).tolist()


def calculate_F3x4(alignment):
    nuc_freq = calculate_frequencies_per_codon_position(alignment)
    return calculate_F3x4_from_nucleotide(alignment.data_type, nuc_freq)


def calculate_substitutions(alignment: Alignment, mapping):
    counts = [0] * (np.max(mapping) + 1)
    data_type = NucleotideDataType(None)
    counter = collections.Counter()

    encoded = [
        list(
            map(
                data_type.encoding,
                map(''.join, zip(*[iter(s.sequence)] * data_type.size)),
            )
        )
        for s in alignment
    ]

    for i, a in enumerate(encoded):
        for j in range(i + 1, len(encoded)):
            counter.update(collections.Counter(list(zip(a, encoded[j]))))

    for tuple in counter:
        if tuple[0] < data_type.state_count and tuple[1] < data_type.state_count:
            counts[mapping[tuple[0]][tuple[1]]] += counter[tuple]
    return counts


def calculate_ts_tv(alignment: Alignment):
    mapping = ((2, 1, 0, 1), (1, 2, 1, 0), (0, 1, 2, 1), (1, 0, 1, 2))
    counts = calculate_substitutions(alignment, mapping)
    return counts[0] / counts[1]


def calculate_kappa(alignment, freqs):
    tstv = calculate_ts_tv(alignment)
    return (tstv * (freqs[0] + freqs[2]) * (freqs[1] + freqs[3])) / (
        (freqs[0] * freqs[2]) + (freqs[1] * freqs[3])
    )
