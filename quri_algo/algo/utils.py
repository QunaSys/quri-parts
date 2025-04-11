from functools import wraps
from time import time
from typing import Callable, ParamSpec, TypeVar

from .interface import AlgorithmResult

T = TypeVar("T", bound=AlgorithmResult)
P = ParamSpec("P")


def timer(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> T:
        t0 = time()
        result = f(*args, **kw)
        t1 = time()
        result.elapsed_time = t1 - t0
        return result

    return wrap
