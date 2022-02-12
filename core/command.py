from typing import NamedTuple, Optional, Callable


class Command(NamedTuple):
    command_bytes: bytes
    response_parser: Optional[Callable] = None
