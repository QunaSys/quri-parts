class Parameter:
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __init__(self, name: str = "") -> None: ...
    @property
    def name(self) -> str: ...