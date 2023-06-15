"""Interface for serializable objects."""
from __future__ import annotations

import abc
import logging
from typing import Any

from .utils import JSONParseError


class JSONSerializable(abc.ABC):
    """Interface making an object JSON serializable.

    Serializable base class establishing
    :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.
    """

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data: dict[str, Any], dic: dict[str, Any]) -> Any:
        """Abstract method to create object from a dictionary.

        :param dict[str, Any] data: dictionary representation of a torchtree object.
        :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
            by their ID.
        :return: torchtree object.
        :rtype: Any
        """
        ...

    @classmethod
    def from_json_safe(cls, data: dict[str, Any], dic: dict[str, Any]) -> Any:
        """Parse dictionary to create object.

        :param dict[str, Any] data: dictionary representation of a torchtree object.
        :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
            by their ID.
        :raises JSONParseError: JSON error
        :return: torchtree object.
        :rtype: Any
        """
        try:
            return cls.from_json(data, dic)
        except JSONParseError as e:
            logging.error(e)
            raise JSONParseError("Calling object with ID '{}'".format(data['id']))
