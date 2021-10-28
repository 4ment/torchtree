import collections

from ..core.model import Identifiable
from ..core.utils import process_objects, register_class


@register_class
class Taxon(Identifiable, collections.UserDict):
    def __init__(self, id_, attributes):
        Identifiable.__init__(self, id_)
        collections.UserDict.__init__(self, attributes)

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"Taxon(id={self.id}, attributes={self.data})"

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        attributes = data.get('attributes', {})
        return cls(id_, attributes)


@register_class
class Taxa(Identifiable, collections.UserList):
    def __init__(self, id_, taxa):
        Identifiable.__init__(self, id_)
        collections.UserList.__init__(self, taxa)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_objects(data['taxa'], dic)
        return cls(id_, taxa)
