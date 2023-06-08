from abc import ABC
from typing import Optional

from .serializable import JSONSerializable


class Identifiable(JSONSerializable, ABC):
    """Abstract class making an object identifiable.

    :param str or None id_: identifier of object
    """

    def __init__(self, id_: Optional[str]) -> None:
        self._id = id_

    @property
    def id(self) -> Optional[str]:
        """Return the identifier."""
        return self._id
