import torch

from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.site_pattern import SitePattern
from torchtree.evolution.taxa import Taxa, Taxon


def test_site_pattern():
    taxa = Taxa(None, [Taxon(taxon, {}) for taxon in 'ABCD'])
    sequences = [
        Sequence(taxon, seq) for taxon, seq in zip('ABCD', ['AAG', 'AAC', 'AAC', 'AAT'])
    ]
    alignment = Alignment(None, sequences, taxa, NucleotideDataType(None))

    site_pattern = SitePattern(None, alignment)
    partials, weights = site_pattern.compute_tips_partials()
    assert torch.all(weights == torch.tensor([[2.0, 1.0]]))

    assert partials[0].shape == torch.Size([4, 2])
