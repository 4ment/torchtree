import torch

from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.site_pattern import SitePattern, compress_alignment
from torchtree.evolution.taxa import Taxa, Taxon


def test_site_pattern():
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in 'ABCD'])
    sequences = [
        Sequence(taxon, seq) for taxon, seq in zip('ABCD', ['AAG', 'AAC', 'AAC', 'AAT'])
    ]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType())
    partials, weights = compress_alignment(alignment)

    site_pattern = SitePattern(None, partials, weights)

    assert torch.all(site_pattern.weights == torch.tensor([[2.0, 1.0]]))

    assert site_pattern.partials[0].shape == torch.Size([4, 2])
