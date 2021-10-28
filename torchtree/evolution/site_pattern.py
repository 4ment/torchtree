from collections import Counter
from typing import List, Optional, Tuple, Union

import torch

from ..core.model import Model
from ..core.utils import process_object, register_class, string_to_list_index
from .alignment import Alignment


@register_class
class SitePattern(Model):
    _tag = 'site_pattern'

    def __init__(
        self, id_: Optional[str], partials: List[torch.Tensor], weights: torch.Tensor
    ) -> None:
        super().__init__(id_)
        self.partials = partials
        self.weights = weights

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        self.weights = self.weights.cuda(device)
        for idx, partial in enumerate(self.partials):
            if partial is None:
                break
            self.partials[idx] = partial.cuda(device)

    def cpu(self) -> None:
        self.weights = self.weights.cpu()
        for idx, partial in enumerate(self.partials):
            if partial is None:
                break
            self.partials[idx] = partial.cpu()

    @property
    def sample_shape(self) -> torch.Size:
        return torch.Size([])

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        alignment = process_object(data['alignment'], dic)
        indices = data.get('indices', None)
        list_of_indices = None
        if indices is not None:
            list_of_indices = [
                string_to_list_index(index_str) for index_str in indices.split(',')
            ]
        partials, weights = compress_alignment(alignment, list_of_indices)

        return cls(id_, partials, weights)


def compress_alignment(
    alignment: Alignment, indices: List[Union[int, slice]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress alignment using data_type.

    :param Alignment alignment: sequence alignment
    :param indices: list of indices: int or slice
    :return: a tuple containing partials and weights
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    taxa, sequences = zip(*alignment)
    if indices is not None:
        sequences_new = [""] * len(sequences)
        for index in indices:
            for idx, sequence in enumerate(sequences):
                sequences_new[idx] += sequence[index]
        count_dict = Counter(list(zip(*sequences_new)))
    else:
        count_dict = Counter(list(zip(*sequences)))
    pattern_ordering = sorted(list(count_dict.keys()))
    patterns_list = list(zip(*pattern_ordering))
    weights = torch.tensor([count_dict[pattern] for pattern in pattern_ordering])
    patterns = dict(zip(taxa, patterns_list))

    partials = []

    for taxon in taxa:
        partials.append(
            torch.tensor(
                [alignment.data_type.partial(c) for c in patterns[taxon]],
                dtype=torch.get_default_dtype(),
            ).t()
        )

    for i in range(len(alignment) - 1):
        partials.append([None] * len(patterns.keys()))
    return partials, weights
