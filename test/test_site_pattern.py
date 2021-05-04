import torch

from phylotorch.evolution.alignment import Alignment, Sequence
from phylotorch.evolution.datatype import NucleotideDataType
from phylotorch.evolution.site_pattern import SitePattern, compress_alignment
from phylotorch.evolution.taxa import Taxon, Taxa


def test_site_pattern():
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in 'ABCD'])
    sequences = [Sequence(taxon, seq) for taxon, seq in zip('ABCD', ['AAG', 'AAC', 'AAC', 'AAT'])]
    alignment = Alignment(None, sequences, taxa)
    partials, weights = compress_alignment(alignment, NucleotideDataType())

    site_pattern = SitePattern(None, partials, weights)

    assert torch.all(site_pattern.weights == torch.tensor([[2., 1.]]))

    assert site_pattern.partials[0].shape == torch.Size([4, 2])
