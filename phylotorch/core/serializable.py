import abc


class JSONSerializable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_json(cls, data, dic):
        ...
