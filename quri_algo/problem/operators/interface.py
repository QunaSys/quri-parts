from typing import Protocol, TypeVar


class OperatorProtocol(Protocol):
    """Represents a generic operator."""

    ...


OperatorT = TypeVar("OperatorT", bound="OperatorProtocol")
