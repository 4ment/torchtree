import collections

from phylotorch.core.model import Identifiable
from phylotorch.core.utils import process_object


Sequence = collections.namedtuple('Sequence', ['taxon', 'sequence'])


class Alignment(Identifiable, collections.UserList):

    def __init__(self, id_, sequences, taxa):
        self._sequence_size = len(sequences[0].sequence)
        self._taxa = taxa
        indexing = {taxon.id: idx for idx, taxon in enumerate(taxa)}
        sequences.sort(key=lambda x: indexing[x.taxon])
        Identifiable.__init__(self, id_)
        collections.UserList.__init__(self, sequences)

    @property
    def sequence_size(self):
        return self._sequence_size

    @property
    def taxa(self):
        return self._taxa

    @classmethod
    def get(cls, id_, filename, taxa):
        sequences = read_fasta_sequences(filename)
        return cls(id_, sequences, taxa)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)

        if 'sequences' in data:
            sequences = []
            for sequence in data['sequences']:
                sequences.append(Sequence(sequence['taxon'], sequence['sequence']))
        elif 'file' in data:
            sequences = read_fasta_sequences(data['file'])
        else:
            raise ValueError('sequences or file should be specified in Alignment')
        return cls(id_, sequences, taxa)


def read_fasta_sequences(filename):
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
