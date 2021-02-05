import numpy as np
import torch
from dendropy import DnaCharacterMatrix

from ..core.model import Model
from ..core.serializable import JSONSerializable


class SitePattern(Model, JSONSerializable):

    def __init__(self, id_, partials, weights):
        self.partials = partials
        self.weights = weights
        super(SitePattern, self).__init__(id_)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        data_type = data['datatype']
        if 'file' in data:
            seqs_args = dict(schema='nexus', preserve_underscores=True)
            with open(data['file']) as fp:
                if next(fp).startswith('>'):
                    seqs_args = dict(schema='fasta')
            if data_type == 'nucleotide':
                alignment = DnaCharacterMatrix.get(path=data['file'], **seqs_args)
        alignment.taxon_namespace.sort()
        partials, weights = get_dna_leaves_partials_compressed(alignment)
        return cls(id_, partials, weights)


def get_dna_leaves_partials_compressed(alignment):
    weights = []
    keep = [True] * alignment.sequence_size

    patterns = {}
    indexes = {}
    for i in range(alignment.sequence_size):
        pat = tuple(alignment[name][i] for name in alignment)

        if pat in patterns:
            keep[i] = False
            patterns[pat] += 1.0
        else:
            patterns[pat] = 1.0
            indexes[i] = pat
    for i in range(alignment.sequence_size):
        if keep[i]:
            weights.append(patterns[indexes[i]])

    partials = []
    dna_map = {'a': [1.0, 0.0, 0.0, 0.0],
               'c': [0.0, 1.0, 0.0, 0.0],
               'g': [0.0, 0.0, 1.0, 0.0],
               't': [0.0, 0.0, 0.0, 1.0]}

    for name in alignment:
        temp = []
        for i, c in enumerate(alignment[name].symbols_as_string()):
            if keep[i]:
                temp.append(dna_map.get(c.lower(), [1., 1., 1., 1.]))
        tip_partials = torch.tensor(np.transpose(np.array(temp)), requires_grad=False)

        partials.append(tip_partials)

    for i in range(len(alignment) - 1):
        partials.append([None] * len(patterns.keys()))
    return partials, torch.tensor(np.array(weights))
