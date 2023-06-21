from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Generic, NamedTuple, TypeVar

from typing_extensions import Concatenate, ParamSpec, TypeAlias

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class RecordableFunctionId(NamedTuple):
    module: str
    qualname: str
    param: Hashable

    def __str__(self) -> str:
        base = f"{self.module}.{self.qualname}"
        if self.param:
            return f"{base}<{str(self.param)}>"
        else:
            return base


class RecordableFunction(Generic[P, R]):
    def __init__(self, f: Callable[P, R], id: RecordableFunctionId):
        self._f = f
        self._id = id

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._f(*args, **kwargs)

    @property
    def id(self) -> RecordableFunctionId:
        return self._id


RecordLevel: TypeAlias = int

INFO: RecordLevel = 20
DEBUG: RecordLevel = 10

_RecKey: TypeAlias = Hashable
_RecValue: TypeAlias = Any
_RecData: TypeAlias = tuple[_RecKey, _RecValue]


class Recorder:
    def __init__(self, fid: RecordableFunctionId) -> None:
        self._func_id = fid

    @contextmanager
    def start_func(self) -> Iterator[None]:
        for session in _active_sessions:
            session.enter_func(self._func_id)
        yield
        for session in _active_sessions:
            session.exit_func(self._func_id)

    def record(self, level: RecordLevel, key: _RecKey, value: _RecValue) -> None:
        for session in _active_sessions:
            if session.is_enabled_for(level, self._func_id):
                session.handler(self._func_id, key, value)

    def debug(self, key: _RecKey, value: _RecValue) -> None:
        self.record(DEBUG, key, value)

    def info(self, key: _RecKey, value: _RecValue) -> None:
        self.record(INFO, key, value)

    def is_enabled_for(self, level: RecordLevel) -> bool:
        return any(
            session.is_enabled_for(level, self._func_id) for session in _active_sessions
        )


_recorders: dict[RecordableFunctionId, Recorder] = {}


def _get_recorder(fid: RecordableFunctionId) -> Recorder:
    if fid in _recorders:
        return _recorders[fid]
    else:
        recorder = Recorder(fid)
        _recorders[fid] = recorder
        return recorder


def recordable(f: Callable[Concatenate[Recorder, P], R]) -> RecordableFunction[P, R]:
    param = ()  # TODO
    f_id = RecordableFunctionId(f.__module__, f.__qualname__, param)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        recorder = _get_recorder(f_id)
        with recorder.start_func():
            return f(recorder, *args, **kwargs)

    rf = RecordableFunction(wrapper, f_id)

    return update_wrapper(rf, f)


@dataclass
class RecordEntry:
    func_id: RecordableFunctionId
    data: _RecData


@dataclass
class RecordGroup:
    func_id: RecordableFunctionId
    entries: list[RecordEntry]

    def add_entry(self, entry: RecordEntry) -> None:
        self.entries.append(entry)


class RecordSet:
    def __init__(self) -> None:
        self._history: list[RecordGroup] = []

    def add_group(self, fid: RecordableFunctionId) -> RecordGroup:
        group = RecordGroup(fid, [])
        self._history.append(group)
        return group

    def remove_last_group(self) -> None:
        self._history.pop()

    def get_history(self, func: RecordableFunction[P, R]) -> Iterable[RecordGroup]:
        return filter(lambda g: g.func_id == func.id, self._history)


class RecordSession:
    def __init__(self) -> None:
        self._levels: dict[RecordableFunctionId, RecordLevel] = {}
        self._record_set = RecordSet()
        self._group_stack: list[RecordGroup] = []

    def set_level(self, level: RecordLevel, func: RecordableFunction[P, R]) -> None:
        self._levels[func.id] = level

    def is_enabled_for(self, level: RecordLevel, fid: RecordableFunctionId) -> bool:
        return fid in self._levels and level >= self._levels[fid]

    def handler(
        self, fid: RecordableFunctionId, key: _RecKey, value: _RecValue
    ) -> None:
        entry = RecordEntry(fid, (key, value))
        self._group_stack[-1].add_entry(entry)

    @contextmanager
    def start(self) -> Iterator[None]:
        _active_sessions.append(self)
        yield
        _active_sessions.pop()

    def enter_func(self, fid: RecordableFunctionId) -> None:
        group = self._record_set.add_group(fid)
        self._group_stack.append(group)

    def exit_func(self, fid: RecordableFunctionId) -> None:
        group = self._group_stack.pop()
        if not group.entries:
            self._record_set.remove_last_group()

    def get_records(self) -> RecordSet:
        return self._record_set


_active_sessions: list[RecordSession] = []
