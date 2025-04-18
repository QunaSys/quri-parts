from functools import wraps
from time import time
from typing import Callable, ParamSpec, Protocol, TypeVar


class HasElapsedTime(Protocol):
    elapsed_time: float | None


T = TypeVar("T", bound=HasElapsedTime)
P = ParamSpec("P")


def timer(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
        t0 = time()
        result = f(*args, **kwargs)
        t1 = time()
        result.elapsed_time = t1 - t0
        return result

    return wrap
