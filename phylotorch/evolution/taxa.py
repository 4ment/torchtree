import collections

from ..core.model import Identifiable
from ..core.utils import process_objects


class Taxon(Identifiable, collections.UserDict):
    def __init__(self, id_, attributes):
        Identifiable.__init__(self, id_)
        collections.UserDict.__init__(self, attributes)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        attributes = data.get('attributes', {})
        return cls(id_, attributes)


class Taxa(Identifiable, collections.UserList):
    def __init__(self, id_, taxa):
        Identifiable.__init__(self, id_)
        collections.UserList.__init__(self, taxa)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_objects(data['taxa'], dic)
        return cls(id_, taxa)
