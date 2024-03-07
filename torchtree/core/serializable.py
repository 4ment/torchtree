"""Interface for serializable objects."""

from __future__ import annotations

import abc
import logging
from typing import Any

from torchtree.core.utils import JSONParseError


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
        except KeyError as e:
            type_ = cls.__name__
            key = e.args[0]
            if key == "id" or "id" not in data:
                raise JSONParseError(f"Missing `id' key for object of type `{type_}'")
            else:
                id_ = data["id"]
                raise JSONParseError(
                    f"Missing key `{key}' for object of type `{type_}' with ID `{id_}'"
                )
        except JSONParseError as e:
            logging.error(e)
            raise JSONParseError(
                "Calling object of type `{}' with ID `{}'".format(
                    cls.__name__, data["id"]
                )
            )
