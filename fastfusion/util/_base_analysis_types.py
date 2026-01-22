from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionKey:
    level: str
    action: str


@dataclass(frozen=True)
class VerboseActionKey(ActionKey):
    tensor: str | None
    einsum: str


@dataclass
class ActionCount:
    total: Any
    max_per_unit: Any

    @staticmethod
    def default():
        return ActionCount(0, 0)
