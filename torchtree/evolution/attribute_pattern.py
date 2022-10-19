from typing import Optional, Union

import torch

from torchtree.core.utils import process_object, register_class

from ..core.model import Model
from .datatype import DataType
from .taxa import Taxa


@register_class
class AttributePattern(Model):
    _tag = 'site_pattern'

    def __init__(
        self, id_: Optional[str], taxa: Taxa, data_type: DataType, attribute: str
    ) -> None:
        super().__init__(id_)
        self.taxa = taxa
        self.data_type = data_type
        self.attribute = attribute

    def compute_tips_states(self):
        return [
            torch.clamp(
                torch.tensor(
                    [self.data_type.encoding(taxon[self.attribute])],
                    dtype=torch.long,
                ),
                max=self.data_type.state_count,
            )
            for taxon in self.taxa
        ], torch.tensor([1.0])

    def compute_tips_partials(self, use_ambiguities=False):
        return [
            torch.tensor(
                [self.data_type.partial(taxon[self.attribute], use_ambiguities)],
                dtype=torch.get_default_dtype(),
            ).t()
            for taxon in self.taxa
        ], torch.tensor([1.0])

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
        taxa = process_object(data['taxa'], dic)
        data_type = process_object(data['data_type'], dic)
        attribute = data['attribute']
        return cls(id_, taxa, data_type, attribute)
