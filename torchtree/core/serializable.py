import abc
import logging

from .utils import JSONParseError


class JSONSerializable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_json(cls, data, dic):
        ...

    @classmethod
    def from_json_safe(cls, data, dic):
        try:
            return cls.from_json(data, dic)
        except JSONParseError as e:
            logging.error(e)
            raise JSONParseError("Calling object with ID '{}'".format(data['id']))
