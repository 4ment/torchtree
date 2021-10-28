from abc import ABC
from typing import Optional

from .serializable import JSONSerializable


class Identifiable(JSONSerializable, ABC):
    def __init__(self, id_: Optional[str]) -> None:
        self._id = id_

    @property
    def id(self) -> Optional[str]:
        return self._id
